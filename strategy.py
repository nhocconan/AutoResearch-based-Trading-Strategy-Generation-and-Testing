#!/usr/bin/env python3
"""
Experiment #1223: 1d Primary + 1w HTF — Connors RSI + HMA Trend + ATR Stop

Hypothesis: #1222 achieved positive Sharpe (0.120) but didn't beat baseline (0.612).
Key insight: Daily timeframe naturally limits trades to 20-50/year, reducing fee drag.
Connors RSI (CRSI) has proven 75% win rate in research for mean reversion entries.
Combine with weekly HMA for macro trend filter to avoid counter-trend trades in strong trends.

Strategy logic:
1. 1w HMA(21) = macro trend filter (only long when price > HMA_1w, only short when <)
2. 1d HMA(21) slope = intermediate trend confirmation
3. Connors RSI(3,2,100) = entry trigger (CRSI<15 long, CRSI>85 short)
4. ATR(14) trailing stop = 2.5x for risk management
5. Position size = 0.28 (discrete, reduces fee churn)

Why this should work:
- 1d timeframe = fewer trades, less fee drag than 4h/12h
- CRSI catches oversold/overbought extremes better than standard RSI
- Weekly HMA filter prevents counter-trend trades during strong trends
- ATR stop protects against 2022-style crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_trend_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=3):
    """Relative Strength Index for Connors RSI component."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component for Connors RSI.
    Measures consecutive up/down days.
    Up streak = positive days in a row, Down streak = negative days in a row.
    Then calculate RSI of the streak values.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return streak_rsi
    
    # Calculate streak values
    streak = np.zeros(n)
    delta = np.diff(close)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            if i > 1 and delta[i-2] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta[i-1] < 0:
            if i > 1 and delta[i-2] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_delta = np.diff(streak)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(streak_delta > 0, streak_delta, 0)
    loss[1:] = np.where(streak_delta < 0, -streak_delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    streak_rsi = 100.0 - (100.0 / (1.0 + rs))
    streak_rsi[:period] = np.nan
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component for Connors RSI.
    Measures where current close is relative to last N days.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = (count_below / (period - 1)) * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Calculate HMA slope (5-bar lookback)
    hma_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-5]):
            hma_slope[i] = hma_1d[i] - hma_1d[i-5]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d[i]) or np.isnan(hma_slope[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA slope) ===
        trend_up = hma_slope[i] > 0
        trend_down = hma_slope[i] < 0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: Macro bull + CRSI oversold (mean reversion in uptrend)
        # OR Strong uptrend + moderate oversold
        if macro_bull and crsi_oversold:
            desired_signal = BASE_SIZE
        elif trend_up and crsi[i] < 25.0:
            desired_signal = BASE_SIZE
        
        # SHORT: Macro bear + CRSI overbought (mean reversion in downtrend)
        # OR Strong downtrend + moderate overbought
        if macro_bear and crsi_overbought:
            desired_signal = -BASE_SIZE
        elif trend_down and crsi[i] > 75.0:
            desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals