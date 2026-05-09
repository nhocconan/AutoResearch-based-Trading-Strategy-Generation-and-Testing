#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for daily EMA and 20 for Donchian/volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above upper Donchian band with volume AND price > daily EMA50 (uptrend)
            if close[i] > upper and vol > 1.5 * vol_ma_val and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower Donchian band with volume AND price < daily EMA50 (downtrend)
            elif close[i] < lower and vol > 1.5 * vol_ma_val and close[i] < ema_1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below lower Donchian band OR trend reverses (price < daily EMA50)
            if close[i] < lower or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above upper Donchian band OR trend reverses (price > daily EMA50)
            if close[i] > upper or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals