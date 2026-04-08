#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels using previous day's range
    # Standard Camarilla: uses previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First value will be NaN due to roll, handled by min_periods later
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r4 = prev_close_1d + range_1d * 1.1 / 2
    camarilla_r3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_1d * 1.1 / 4
    camarilla_s4 = prev_close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily trend: EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h[i]) or np.isnan(camarilla_r4_12h[i]) or
            np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or
            np.isnan(camarilla_s4_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S3 or trend fails (price < EMA50)
            if close[i] < camarilla_s3_12h[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R3 or trend fails (price > EMA50)
            if close[i] > camarilla_r3_12h[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price > R4 + above EMA50 + volume
            if (close[i] > camarilla_r4_12h[i] and 
                close[i] > ema_50_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S4 + below EMA50 + volume
            elif (close[i] < camarilla_s4_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals