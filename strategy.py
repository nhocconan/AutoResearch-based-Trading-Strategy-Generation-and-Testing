#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeTrend_V1"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-day)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 4h trend filter: EMA(34)
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average (20 * 4h = 80h ~ 3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_34[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter
        uptrend = price > ema
        downtrend = price < ema
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and uptrend
            if price > high_20_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and downtrend
            elif price < low_20_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below 20-day low or reverse signal
            if price < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < low_20_aligned[i] and volume_ok and downtrend:
                # Reverse to short
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above 20-day high or reverse signal
            if price > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > high_20_aligned[i] and volume_ok and uptrend:
                # Reverse to long
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals