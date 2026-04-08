#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n1 = len(close_1d)
    camarilla_s3 = np.full(n1, np.nan)
    camarilla_r3 = np.full(n1, np.nan)
    camarilla_s4 = np.full(n1, np.nan)
    camarilla_r4 = np.full(n1, np.nan)
    
    for i in range(1, n1):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_s3[i] = prev_close - 1.0 * range_val * 1.250
        camarilla_r3[i] = prev_close + 1.0 * range_val * 1.250
        camarilla_s4[i] = prev_close - 1.0 * range_val * 1.625
        camarilla_r4[i] = prev_close + 1.0 * range_val * 1.625
    
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # 1d trend: 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 6h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or trend fails
            if close[i] < camarilla_s3_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or trend fails
            if close[i] > camarilla_r3_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_6h[i]
            bearish = close[i] < ema_50_6h[i]
            
            # Long: break above R4 with volume (continuation in uptrend)
            if (close[i] > camarilla_r4_6h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below S4 with volume (continuation in downtrend)
            elif (close[i] < camarilla_s4_6h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Long: bounce from S3 in uptrend
            elif (close[i] > camarilla_s3_6h[i] and 
                  close[i-1] <= camarilla_s3_6h[i-1] and
                  bullish and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bounce from R3 in downtrend
            elif (close[i] < camarilla_r3_6h[i] and 
                  close[i-1] >= camarilla_r3_6h[i-1] and
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals