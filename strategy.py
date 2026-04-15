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
    
    # Get daily HTF data once before loop (6h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    prev_close = np.concatenate([[daily_close[0]], daily_close[:-1]])
    prev_high = np.concatenate([[daily_high[0]], daily_high[:-1]])
    prev_low = np.concatenate([[daily_low[0]], daily_low[:-1]])
    
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align HTF indicators to 6h timeframe with proper delay
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_6h[i]) or np.isnan(camarilla_s3_6h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h Donchian breakout: price breaks above/below 20-period channel
        # 2. Volume confirmation: volume > 1.8x average
        # 3. ATR filter: volatility > 0.5 * price (avoid low volatility chop)
        # 4. Camarilla filter: avoid fading at R3/S3 levels
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Donchian breakout with volume, not at resistance
        if (close[i] > highest_20[i] and              # Donchian breakout
            volume_ratio[i] > 1.8 and                 # Volume confirmation
            atr[i] > 0.005 * close[i] and             # Sufficient volatility
            close[i] < camarilla_r3_6h[i] * 1.02):    # Not too close to R3 (avoid fading)
            signals[i] = 0.25
            
        # Short conditions: Donchian breakdown with volume, not at support
        elif (close[i] < lowest_20[i] and             # Donchian breakdown
              volume_ratio[i] > 1.8 and               # Volume confirmation
              atr[i] > 0.005 * close[i] and           # Sufficient volatility
              close[i] > camarilla_s3_6h[i] * 0.98):  # Not too close to S3 (avoid fading)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Breakout_Volume_ATR_Camarilla_Filter"
timeframe = "6h"
leverage = 1.0