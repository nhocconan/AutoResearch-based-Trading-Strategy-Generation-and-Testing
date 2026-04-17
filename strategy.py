#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams Alligator ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Alligator lines: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    # SMMA = smoothed moving average (similar to EMA with alpha = 1/period)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) == 0:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # === 6h Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 13:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = high[i]
            lowest_low[i] = low[i]
    
    williams_r = np.full_like(close, np.nan)
    for i in range(len(close)):
        if not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]) and highest_high[i] != lowest_low[i]:
            williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
        else:
            williams_r[i] = np.nan
    
    # === 1d Williams %R for trend filter ===
    highest_high_1d = np.full_like(close_1d, np.nan)
    lowest_low_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 13:
            highest_high_1d[i] = np.max(high_1d[i-13:i+1])
            lowest_low_1d[i] = np.min(low_1d[i-13:i+1])
        elif i > 0:
            highest_high_1d[i] = np.max(high_1d[0:i+1])
            lowest_low_1d[i] = np.min(low_1d[0:i+1])
        else:
            highest_high_1d[i] = high_1d[i]
            lowest_low_1d[i] = low_1d[i]
    
    williams_r_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(highest_high_1d[i]) and not np.isnan(lowest_low_1d[i]) and highest_high_1d[i] != lowest_low_1d[i]:
            williams_r_1d[i] = ((highest_high_1d[i] - close_1d[i]) / (highest_high_1d[i] - lowest_low_1d[i])) * -100
        else:
            williams_r_1d[i] = np.nan
    
    # === Align indicators to 6h timeframe ===
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 6h Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        # Williams %R: > -20 = overbought, < -80 = oversold
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Alligator aligned up + Williams %R oversold + 1d trend up (Williams %R > -50)
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                williams_r[i] < -80 and 
                williams_r_1d_aligned[i] > -50 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Alligator aligned down + Williams %R overbought + 1d trend down (Williams %R < -50)
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  williams_r[i] > -20 and 
                  williams_r_1d_aligned[i] < -50 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Williams %R overbought
            if (lips_aligned[i] < teeth_aligned[i] or 
                williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Williams %R oversold
            if (lips_aligned[i] > teeth_aligned[i] or 
                williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Alligator_WilliamsR_Trend_v1"
timeframe = "6h"
leverage = 1.0