#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # Calculate 20-period Donchian channels on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for Donchian and volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_30_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_30_1d_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above upper Donchian with volume AND price > 1d EMA30 (uptrend)
            if close[i] > upper_channel and vol > 2.0 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower Donchian with volume AND price < 1d EMA30 (downtrend)
            elif close[i] < lower_channel and vol > 2.0 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below lower Donchian OR trend reverses (price < 1d EMA30)
            if close[i] < lower_channel or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above upper Donchian OR trend reverses (price > 1d EMA30)
            if close[i] > upper_channel or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals