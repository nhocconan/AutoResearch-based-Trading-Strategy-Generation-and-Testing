#!/usr/bin/env python3
name = "12h_Donchian_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channels (20-period)
    upper_donchian = np.full_like(high_12h, np.nan)
    lower_donchian = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        upper_donchian[i] = np.max(high_12h[i-20:i])
        lower_donchian[i] = np.min(low_12h[i-20:i])
    
    # Align to 12h
    upper_donchian_aligned = align_htf_to_ltf(prices, df_12h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_12h, lower_donchian)
    
    # Volume filter: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above daily EMA200 trend
            if (close[i] > upper_donchian_aligned[i] and 
                volume_surge and 
                close[i] > ema_200_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and below daily EMA200 trend
            elif (close[i] < lower_donchian_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_200_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band
            if position == 1:
                # Exit long: price touches or goes below lower band
                if close[i] <= lower_donchian_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above upper band
                if close[i] >= upper_donchian_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals