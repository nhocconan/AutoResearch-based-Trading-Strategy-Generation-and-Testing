#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Momentum_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # 6h momentum: RSI(14) with proper min_periods
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above R1 with volume spike and RSI > 50
            long_cond = (close[i] > r1_aligned[i] and vol_spike[i] and rsi_values[i] > 50)
            
            # Short entry: price crosses below S1 with volume spike and RSI < 50
            short_cond = (close[i] < s1_aligned[i] and vol_spike[i] and rsi_values[i] < 50)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot or RSI < 40
            if close[i] < pivot_aligned[i] or rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot or RSI > 60
            if close[i] > pivot_aligned[i] or rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points provide key support/resistance levels. 
# Price breaking above R1 with momentum (RSI>50) and volume confirmation indicates bullish momentum.
# Price breaking below S1 with momentum (RSI<50) and volume confirmation indicates bearish momentum.
# Exits when price returns to pivot or momentum fades (RSI<40 for longs, >60 for shorts).
# Weekly timeframe ensures alignment with major trend, reducing counter-trend trades.
# Designed for 6h timeframe to balance trade frequency (target: 15-30 trades/year) and responsiveness.
# Works in both bull and bear markets by following momentum with volume confirmation.