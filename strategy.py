#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h: Donchian channels (20) for trend ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Donchian trend: 1 if above upper band, -1 if below lower band, 0 otherwise
    donchian_trend = np.zeros(len(high_12h))
    donchian_trend[high_12h > high_20] = 1
    donchian_trend[low_12h < low_20] = -1
    
    # Align to 4h timeframe
    donchian_trend_aligned = align_htf_to_ltf(prices, df_12h, donchian_trend)
    
    # === 4h: Price, volume, and Donchian breakout (20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20) for breakout
    high_20_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_trend_aligned[i]) or 
            np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donchian_trend_val = donchian_trend_aligned[i]
        vol_ratio_val = vol_ratio[i]
        upper_20_4h = high_20_4h[i]
        lower_20_4h = low_20_4h[i]
        
        if position == 0:
            # Long: 4h Donchian breakout up + 12h uptrend + volume confirmation
            if (high_val > upper_20_4h and 
                donchian_trend_val > 0 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: 4h Donchian breakdown down + 12h downtrend + volume confirmation
            elif (low_val < lower_20_4h and 
                  donchian_trend_val < 0 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to lower Donchian band or trend changes
            if close_val < lower_20_4h or donchian_trend_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to upper Donchian band or trend changes
            if close_val > upper_20_4h or donchian_trend_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals