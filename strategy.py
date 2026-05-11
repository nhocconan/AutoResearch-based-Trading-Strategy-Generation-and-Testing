#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Previous day's values
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla R3, S3 levels
    range_val = prev_high - prev_low
    R3 = prev_close + range_val * 1.1
    S3 = prev_close - range_val * 1.1
    
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1D EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio (24-period average for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: Price breaks above R3 with volume surge and above EMA34/uptrend
            if (close[i] > R3_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S3 with volume surge and below EMA34/downtrend
            elif (close[i] < S3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price returns below EMA34 or closes below S3
                if (close[i] < ema_34_aligned[i]) or (close[i] < S3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Exit short: Price returns above EMA34 or closes above R3
                if (close[i] > ema_34_aligned[i]) or (close[i] > R3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals