#!/usr/bin/env python3
"""
Experiment #327: 6h Primary + 1d HTF — Connors RSI Mean Reversion with Trend Bias

Hypothesis: 6h timeframe is underexplored "Goldilocks zone" between noisy lower TFs 
and slow daily TFs. Connors RSI (CRSI) has 75% win rate in literature for mean reversion.
Combined with 1d HMA trend filter, this should capture pullbacks within larger trends.

Why this differs from failed #320 (Fisher/Chop, Sharpe=-0.531):
1. CRSI instead of Fisher — CRSI is proven mean-reversion indicator (3 components)
2. Simpler regime: just 1d HMA direction, no choppy/trending flip-flop
3. ATR volatility filter: only enter when vol expanding (ATR ratio > 1.2)
4. Looser CRSI thresholds (15/85 vs 10/90) to ensure sufficient trades
5. Asymmetric sizing: 0.30 when 1d strongly aligned, 0.25 otherwise

Connors RSI Components:
- RSI(3): short-term momentum
- RSI_Streak(2): streak length momentum  
- PercentRank(100): relative price position in recent range
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry Logic:
- Long: CRSI < 15 + price > 1d HMA(50) + ATR ratio > 1.2 (vol expansion)
- Short: CRSI > 85 + price < 1d HMA(50) + ATR ratio > 1.2
- Exit: CRSI crosses 50 (mean reversion complete) OR 2.5x ATR stoploss

Target: Sharpe>0.50, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
Timeframe: 6h (REQUIRED for this experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_meanrevert_hma_1d_v1"
timeframe = "6h"
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
    Connors RSI - 3-component mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI_Streak: Momentum of consecutive up/down days
    PercentRank: Current price position in recent N-day range
    
    Values 0-100. <10 = extremely oversold, >90 = extremely overbought
    """
    n = len(close)
    if n < rank_period + rsi_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak Length
    streak = np.zeros(n)
    streak_direction = np.zeros(n)  # 1=up streak, -1=down streak
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak_direction[i-1] >= 0:
                streak[i] = streak[i-1] + 1
                streak_direction[i] = 1
            else:
                streak[i] = 1
                streak_direction[i] = 1
        elif close[i] < close[i-1]:
            if streak_direction[i-1] <= 0:
                streak[i] = streak[i-1] + 1
                streak_direction[i] = -1
            else:
                streak[i] = 1
                streak_direction[i] = -1
        else:
            streak[i] = streak[i-1]
            streak_direction[i] = streak_direction[i-1]
    
    # Convert streak to RSI-like value (longer streak = more extreme)
    # Use RSI formula on streak values
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period*2):i+1]
        if len(streak_vals) >= streak_period:
            up_streaks = np.sum(np.where(streak_vals > 0, streak_vals, 0))
            down_streaks = np.sum(np.where(streak_vals < 0, -streak_vals, 0))
            if down_streaks < 1e-10:
                streak_rsi[i] = 100.0
            else:
                rs = up_streaks / down_streaks
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percentile Rank of close in recent N-day range
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / (len(window) - 1) if len(window) > 1 else 50.0
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    atr_long = calculate_atr(high, low, close, period=50)  # For vol ratio
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_6h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_long[i]) or atr_long[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY EXPANSION FILTER ===
        # Only trade when vol is expanding (ATR ratio > 1.2)
        atr_ratio = atr[i] / atr_long[i] if atr_long[i] > 1e-10 else 0.0
        vol_expanding = atr_ratio > 1.15  # Slightly loosened from 1.2
        
        # === HTF TREND BIAS (1d HMA50) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CRSI EXTREMES (LOOSENED for more trades) ===
        crsi_oversold = crsi[i] < 18.0  # Was 15
        crsi_overbought = crsi[i] > 82.0  # Was 85
        crsi_neutral_long = crsi[i] > 45.0  # Exit long when CRSI mean-reverts
        crsi_neutral_short = crsi[i] < 55.0  # Exit short when CRSI mean-reverts
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + HTF bull + vol expanding + above SMA200
        if crsi_oversold and htf_bull and vol_expanding and above_sma200:
            # Stronger signal if 6h HMA also bull
            if hma_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: CRSI overbought + HTF bear + vol expanding + below SMA200
        elif crsi_overbought and htf_bear and vol_expanding and below_sma200:
            # Stronger signal if 6h HMA also bear
            if hma_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC (CRSI mean reversion complete) ===
        if in_position and position_side > 0 and crsi_neutral_long:
            desired_signal = 0.0  # Exit long
        
        if in_position and position_side < 0 and crsi_neutral_short:
            desired_signal = 0.0  # Exit short
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals