#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (previous day's values)
    range_1d = high_1d - low_1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: R4 = close + 1.5*range, R3 = close + 1.1*range, etc.
    r4_1d = close_1d + 1.5 * range_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Align pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 12h trend: 34-period EMA (responsive but smooth)
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34[i]) or np.isnan(pivot_12h[i]) or np.isnan(r4_12h[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S3 or trend fails
            if close[i] < s3_12h[i] or close[i] < ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price > R3 or trend fails
            if close[i] > r3_12h[i] or close[i] > ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34[i]
            bearish = close[i] < ema_34[i]
            
            # Long: price > R4 + bullish trend + volume
            if (close[i] > r4_12h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.30
            # Short: price < S4 + bearish trend + volume
            elif (close[i] < s4_12h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.30
    
    return signals