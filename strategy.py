#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3_S3_Fade_Volume_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 4.0  # R3 = C + 1.1*(H-L)/4
    s3_1d = close_1d - range_1d * 1.1 / 4.0  # S3 = C - 1.1*(H-L)/4
    r4_1d = close_1d + range_1d * 1.1 / 2.0  # R4 = C + 1.1*(H-L)/2
    s4_1d = close_1d - range_1d * 1.1 / 2.0  # S4 = C - 1.1*(H-L)/2
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx[i]
        r3 = r3_6h[i]
        s3 = s3_6h[i]
        r4 = r4_6h[i]
        s4 = s4_6h[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        trending = adx_val > 20  # Moderate trend filter
        
        if position == 0:
            # Fade at R3/S3: Short at R3, Long at S3 with volume and trend
            if price > r3 and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
            elif price < s3 and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: Price reaches S4 (strong support) or reverses below R3
            if price < s4 or price < r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches R4 (strong resistance) or reverses above S3
            if price > r4 or price > s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals