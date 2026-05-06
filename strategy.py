#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1, S1) with 6h EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R1 AND 6h EMA50 > EMA50 previous (uptrend) AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla S1 AND 6h EMA50 < EMA50 previous (downtrend) AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back through 1d Camarilla pivot point (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels from higher timeframe (1d) provide strong support/resistance
# 6h EMA50 trend filter ensures we trade with the intermediate trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "4h_Camarilla_R1S1_Breakout_6hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed 1d bars for pivot (HLC of previous day)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (using previous day's HLC)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 6h data ONCE before loop for EMA50 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need at least 50 completed 6h bars for EMA50
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA50
    ema_50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_50_6h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 6h EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > r1_aligned[i] and 
                ema_50_6h_aligned[i] > ema_50_6h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, 6h EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < s1_aligned[i] and 
                  ema_50_6h_aligned[i] < ema_50_6h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below pivot point (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above pivot point (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals