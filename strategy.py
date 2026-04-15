#!/usr/bin/env python3
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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Based on previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    prev_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    r4 = camarilla_pp + (camarilla_range * 1.1 / 2)
    r3 = camarilla_pp + (camarilla_range * 1.1 / 4)
    r2 = camarilla_pp + (camarilla_range * 1.1 / 6)
    r1 = camarilla_pp + (camarilla_range * 1.1 / 12)
    s1 = camarilla_pp - (camarilla_range * 1.1 / 12)
    s2 = camarilla_pp - (camarilla_range * 1.1 / 6)
    s3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    s4 = camarilla_pp - (camarilla_range * 1.1 / 2)
    
    # Align HTF indicators to 6h timeframe with proper delay
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h Donchian channels (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 6h price breaks above R1 AND above 20-period Donchian high with volume confirmation
        # Short: 6h price breaks below S1 AND below 20-period Donchian low with volume confirmation
        # Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above R1 with Donchian confirmation and volume
        if (close[i] > r1_6h[i] and            # Price above Camarilla R1
            close[i] > highest_20[i] and       # Price above 20-period Donchian high (breakout confirmation)
            volume_ratio[i] > 1.5):            # Volume confirmation (1.5x average)
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S1 with Donchian confirmation and volume
        elif (close[i] < s1_6h[i] and          # Price below Camarilla S1
              close[i] < lowest_20[i] and      # Price below 20-period Donchian low (breakdown confirmation)
              volume_ratio[i] > 1.5):          # Volume confirmation (1.5x average)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1_S1_Donchian_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0