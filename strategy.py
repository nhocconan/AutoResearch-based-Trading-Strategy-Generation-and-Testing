#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels (R3, S3) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Camarilla formula: R3 = Close + 1.1*(High-Low)/2, S3 = Close - 1.1*(High-Low)/2
    camarilla_r3 = close_1d_arr + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d_arr - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above R3 with volume and above 1d EMA trend
            if close[i] > camarilla_r3_aligned[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume and below 1d EMA trend
            elif close[i] < camarilla_s3_aligned[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals