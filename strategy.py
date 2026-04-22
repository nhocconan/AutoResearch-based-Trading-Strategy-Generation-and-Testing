#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction filter + RSI(14) + chop regime filter (14)
# Uses weekly trend to filter 1d signals: only take long when weekly KAMA is rising,
# short when weekly KAMA is falling. On 1d, enter long when RSI < 30 and chop < 50,
# short when RSI > 70 and chop < 50. Chop > 50 avoids trading in ranging markets.
# Designed to work in bull/bear via trend filter and mean reversion in choppy markets.
# Targets 10-25 trades/year with low turnover.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for KAMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on weekly
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0)  # placeholder, will fix below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        volatility[i] = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Alternative simpler: use EMA as proxy for trend (more stable)
    # But per instructions, use proper KAMA - however, for robustness, use EMA(10) trend
    # Actually, let's use a simple but correct adaptive method: KAMA with fixed lookback
    window = 10
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    # Volatility: sum of absolute changes over last 'window' periods
    volatility = np.zeros_like(close_1w)
    for i in range(window, len(close_1w)):
        volatility[i] = np.sum(np.abs(np.diff(close_1w[i-window:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # KAMA slope: rising if today > yesterday
    kama_slope = np.diff(kama, prepend=0)  # positive = rising
    
    # Align KAMA slope to daily
    kama_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_slope)
    
    # Load daily data for RSI and chop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing: alpha = 1/period
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 0  # not enough data
    
    # Calculate Choppiness Index(14) on daily
    # TR = True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = np.zeros_like(close_1d)
    for i in range(13, len(tr)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_1d)
    lowest_low = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if highest_high[i] - lowest_low[i] != 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # undefined, set to neutral
    
    # Align RSI and chop to daily (they are already daily, but align for consistency)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for indicators
    start_idx = max(14, 13)  # RSI and chop need 14 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
        
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        kama_trend = kama_slope_aligned[i]  # >0 = rising, <0 = falling
        
        # Only trade when chop < 50 (not ranging)
        if chop_val >= 50:
            # In ranging market, stay flat
            signals[i] = 0.0
            continue
        
        # In trending/choppy but not ranging market:
        # Long when RSI < 30 (oversold) and weekly KAMA rising
        # Short when RSI > 70 (overbought) and weekly KAMA falling
        if rsi_val < 30 and kama_trend > 0:
            signals[i] = 0.25
        elif rsi_val > 70 and kama_trend < 0:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0