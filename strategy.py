#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Camarilla R3/S3 levels from 1d HTF with 1w trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 1w EMA34 is rising AND 6h volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 1w EMA34 is falling AND 6h volume > 1.5 * avg_volume(20)
# Exit when price returns to 1d Camarilla midpoint (Pivot) or opposite extreme (R4/S4)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# 1d Camarilla provides mathematically derived support/resistance levels proven effective
# 1w EMA34 ensures we trade with the weekly trend while reducing noise on 6h timeframe
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "6h_1dCamarilla_R3S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "6h"
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
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla formulas: Pivot = (H+L+C)/3, Range = H-L
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels: R3 = Pivot + Range * 1.1/2, R4 = Pivot + Range * 1.1
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    # Support levels: S3 = Pivot - Range * 1.1/2, S4 = Pivot - Range * 1.1
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, EMA34 rising, volume spike
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, EMA34 falling, volume spike
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot or below S4 (strong reversal)
            if close[i] <= pivot_1d_aligned[i] or close[i] <= s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot or above R4 (strong reversal)
            if close[i] >= pivot_1d_aligned[i] or close[i] >= r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals