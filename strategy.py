#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Volume_Spike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_20d_4h = align_htf_to_ltf(prices, df_1d, upper_20d)
    lower_20d_4h = align_htf_to_ltf(prices, df_1d, lower_20d)
    
    # Volume spike: current volume > 1.8x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 50-period EMA on 4h close
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        if np.isnan(upper_20d_4h[i]) or np.isnan(lower_20d_4h[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_50[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume spike and above EMA50
            if price > upper_20d_4h[i] and volume_spike and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume spike and below EMA50
            elif price < lower_20d_4h[i] and volume_spike and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 20-day low (reversal signal)
            if price < lower_20d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 20-day high (reversal signal)
            if price > upper_20d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals