#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R1 (first resistance level) in 1d uptrend with volume > 1.8x 20-period MA.
Short when price breaks below S1 (first support level) in 1d downtrend with volume > 1.8x 20-period MA.
Exit when price reverts to the 1d EMA34 or opposite Camarilla level (S1 for longs, R1 for shorts).
Camarilla R1/S1 levels provide tighter stoploss and better risk/reward than R3/S3, reducing whipsaw.
Designed for moderate trade frequency (~50-100/year) with strong edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1/S1 = close ± 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 1.8x 20-period MA (confirmation)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend AND volume spike
            if close[i] > r1_1d_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND downtrend AND volume spike
            elif close[i] < s1_1d_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reverts to 1d EMA34 or opposite Camarilla level
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below EMA34 or below S1 (opposite level)
                if close[i] < ema_34_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above EMA34 or above R1 (opposite level)
                if close[i] > ema_34_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0