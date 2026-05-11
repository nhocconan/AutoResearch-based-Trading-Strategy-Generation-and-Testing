#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Previous day high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels R3 and S3 (outer bands for breakout)
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume filter: 20-period average
    df_1d_vol = get_htf_data(prices, '1d')['volume'].values
    vol_ma_1d = pd.Series(df_1d_vol).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_ratio = df_1d_vol / vol_ma_1d
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check session
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume and above 4h EMA50 trend
            if (close[i] > r3_aligned[i] and 
                volume_surge and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 with volume and below 4h EMA50 trend
            elif (close[i] < s3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to opposite S1/R1 level (inner band for mean reversion)
            if position == 1:
                # Calculate S1 for exit
                s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
                s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
                if close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Calculate R1 for exit
                r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
                r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
                if close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals