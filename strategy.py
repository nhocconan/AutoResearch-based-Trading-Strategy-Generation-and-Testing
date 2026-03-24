#!/usr/bin/env python3
"""
Experiment #592: 12h Primary + 1d HTF — Connors RSI Mean Reversion + HMA Trend Filter

Hypothesis: 12h timeframe with Connors RSI (CRSI) provides superior entry timing for
mean reversion in trending markets. CRSI combines 3 components for robust oversold/overbought
detection. Combined with 1d HMA for trend bias, this should capture pullbacks in trends
while avoiding counter-trend trades.

Key differences from failed experiments:
1. Connors RSI instead of standard RSI — better for short-term extremes
2. Simpler regime logic — trend filter only, no complex chop/ADX switching
3. Looser entry thresholds — CRSI<15/>85 instead of <10/>90 to ensure trades
4. 12h primary TF — proven to work better than 6h/4h in recent tests
5. Discrete signal sizes with proper stoploss

Strategy logic:
1. 1d HMA(21) = trend bias (long only above, short only below)
2. 12h HMA(21) = primary trend confirmation
3. 12h Connors RSI(3,2,100) = entry timing (oversold/overbought)
4. 12h ATR(14)*2.5 = stoploss on all positions
5. Exit when CRSI crosses mid-range (50) or stoploss hit

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_trend_1d_v1"
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

def calculate_streak_rsi(close, period=2):
    """
    Streak RSI component of Connors RSI
    Measures RSI of consecutive up/down streak length
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on absolute streak values
    streak_rsi = calculate_rsi(np.abs(streak), period)
    # Invert: long streak up = high RSI, but we want it to signal overbought
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures current price change percentile vs last N periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1] if i > 0 else 0
        changes = []
        for j in range(i - period + 1, i + 1):
            if j > 0:
                changes.append(close[j] - close[j-1])
        
        if len(changes) > 0:
            count_below = sum(1 for c in changes if c < current_change)
            pct_rank[i] = 100.0 * count_below / len(changes)
        else:
            pct_rank[i] = 50.0
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Values < 10 = oversold (long opportunity)
    Values > 90 = overbought (short opportunity)
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Looser than 10 to ensure trades
        crsi_overbought = crsi[i] > 85.0  # Looser than 90 to ensure trades
        crsi_neutral = crsi[i] >= 40.0 and crsi[i] <= 60.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Oversold CRSI + uptrend confirmation
        if trend_bull and hma_bull and crsi_oversold:
            desired_signal = SIZE_ENTRY
        
        # SHORT: Overbought CRSI + downtrend confirmation
        elif trend_bear and hma_bear and crsi_overbought:
            desired_signal = -SIZE_ENTRY
        
        # EXIT: CRSI returns to neutral (take profit)
        if in_position and crsi_neutral:
            desired_signal = SIZE_EXIT
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = SIZE_EXIT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_ENTRY * 0.9:
            final_signal = SIZE_ENTRY
        elif desired_signal <= -SIZE_ENTRY * 0.9:
            final_signal = -SIZE_ENTRY
        else:
            final_signal = SIZE_EXIT
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals