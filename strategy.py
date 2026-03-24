#!/usr/bin/env python3
"""
Experiment #143: 6h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Trend Bias

Hypothesis: After 142 failed experiments, the key insight for 6h timeframe:
- 6h is unexplored territory between 4h and 12h - sweet spot for multi-day swings
- Connors RSI (CRSI) excels at catching pullbacks in established trends (75% win rate literature)
- 1w HMA provides major trend bias (bull/bear market filter)
- 1d HMA provides intermediate trend confirmation
- 6h CRSI < 25 in uptrend = long entry (oversold pullback)
- 6h CRSI > 75 in downtrend = short entry (overbought rally)
- Choppiness Index filters out range-bound markets where mean reversion fails
- LOOSE entry thresholds ensure >=30 trades on train, >=3 on test for ALL symbols

Key design choices:
- Timeframe: 6h (target 30-60 trades/year, middle ground)
- HTF: 1w HMA(50) for major trend, 1d HMA(21) for intermediate trend
- Entry: CRSI extremes (25/75) aligned with HTF trend direction
- Filter: CHOP < 55 to avoid choppy markets
- Position size: 0.28 (28% of capital, conservative per Rule 4)
- Stoploss: 2.5x ATR trailing stop
- Minimal filters to ensure trade generation on BTC/ETH/SOL

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_hma_chop_1d1w_v1"
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

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI Component 2: Streak RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = 1.0 if np.isnan(streak[i-1]) or streak[i-1] <= 0 else streak[i-1] + 1.0
        elif close[i] < close[i-1]:
            streak[i] = -1.0 if np.isnan(streak[i-1]) or streak[i-1] >= 0 else streak[i-1] - 1.0
        else:
            streak[i] = 0.0
    
    # RSI of streak values
    streak_series = pd.Series(streak)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        window = streak[max(0, i-period):i+1]
        if len(window) >= period:
            gains = np.sum(np.where(window > 0, window, 0))
            losses = np.sum(np.where(window < 0, -window, 0))
            if losses < 1e-10:
                streak_rsi[i] = 100.0
            else:
                rs = gains / losses
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Component 3: Percent Rank
    Percentage of prior closes lower than current close
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period:i]
        count_lower = np.sum(window < close[i])
        pr[i] = 100.0 * count_lower / period
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate reversal opportunities
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(len(close))
    crsi[:] = np.nan
    
    for i in range(len(close)):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use CHOP < 55 as filter to avoid mean reversion in choppy markets
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (Rule 4: max 0.40)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA - Major Trend) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF CONFIRMATION (1d HMA - Intermediate Trend) ===
        htf_confirm_bull = close[i] > hma_1d_aligned[i]
        htf_confirm_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS FILTER ===
        # Only trade when CHOP < 55 (not too choppy for mean reversion)
        chop_ok = chop[i] < 55.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Long: CRSI < 30 (oversold pullback in uptrend)
        # Short: CRSI > 70 (overbought rally in downtrend)
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + HTF bull + 1d confirm + chop filter + 6h HMA bull
        if crsi_oversold and htf_bull and htf_confirm_bull and chop_ok and hma_bull:
            desired_signal = SIZE
        
        # SHORT: CRSI overbought + HTF bear + 1d confirm + chop filter + 6h HMA bear
        elif crsi_overbought and htf_bear and htf_confirm_bear and chop_ok and hma_bear:
            desired_signal = -SIZE
        
        # FALLBACK 1: Strong CRSI signal with 1w trend only (ignore 1d)
        elif crsi_oversold and htf_bull and chop_ok and crsi[i] < 20.0:
            desired_signal = SIZE * 0.7
        elif crsi_overbought and htf_bear and chop_ok and crsi[i] > 80.0:
            desired_signal = -SIZE * 0.7
        
        # FALLBACK 2: Very extreme CRSI (catch major reversals)
        elif crsi[i] < 15.0 and chop_ok:
            desired_signal = SIZE * 0.5
        elif crsi[i] > 85.0 and chop_ok:
            desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES (Rule 4) ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
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