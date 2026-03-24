#!/usr/bin/env python3
"""
Experiment #072: 12h Primary + 1d HTF — Connors RSI + HMA Trend + Vol Filter

Hypothesis: After 71 failed experiments, the pattern is clear:
- Complex multi-filter strategies fail (too many conditions = 0 trades or negative Sharpe)
- 12h timeframe needs simpler logic with fewer confluence requirements
- Connors RSI has proven 75% win rate for mean reversion
- 1d HMA provides clean trend bias without over-filtering
- ATR vol filter prevents entries during panic spikes

Key design:
- Connors RSI(3,2,100) for entry timing
- 1d HMA(50) for major trend direction
- ATR ratio filter (ATR7/ATR30 < 2.0) to avoid vol spikes
- Position size: 0.30 (30% of capital)
- Stoploss: 3x ATR trailing
- Target: 30-60 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very fast RSI for short-term momentum
    RSI_Streak(2): RSI of up/down streak length
    PercentRank(100): Percentile rank of price change over 100 periods
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - calculate streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    # RSI of streak - use streak direction as "price"
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_streak = max(0, streak[i])
        down_streak = max(0, -streak[i])
        
        # Simple streak RSI approximation
        if i < streak_period + 10:
            streak_rsi[i] = 50.0
        else:
            # Count up vs down streaks in lookback
            up_count = np.sum(streak[max(0, i-streak_period*5):i] > 0)
            down_count = np.sum(streak[max(0, i-streak_period*5):i] < 0)
            total = up_count + down_count
            if total > 0:
                streak_rsi[i] = 100.0 * up_count / total
            else:
                streak_rsi[i] = 50.0
    
    # Percent Rank - percentile of current change vs last rank_period changes
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        changes = np.diff(close[max(0, i-rank_period):i+1])
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            rank = np.sum(changes < current_change)
            percent_rank[i] = 100.0 * rank / len(changes)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    atr7 = calculate_atr(high, low, close, period=7)
    atr30 = calculate_atr(high, low, close, period=30)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr7[i]) or np.isnan(atr30[i]) or atr30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Avoid entering during vol spikes (ATR7/ATR30 > 2.0 = panic)
        vol_ratio = atr7[i] / atr30[i]
        vol_ok = vol_ratio < 2.5  # Allow some vol, just not extreme
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === DESIRED SIGNAL (Simpler logic for more trades) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + HTF not bearish + vol ok
        if crsi_oversold and vol_ok:
            if htf_bull or hma_bull:
                desired_signal = SIZE
            elif not htf_bear:
                desired_signal = SIZE * 0.5
        
        # SHORT: CRSI overbought + HTF not bullish + vol ok
        if crsi_overbought and vol_ok:
            if htf_bear or hma_bear:
                desired_signal = -SIZE
            elif not htf_bull:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals