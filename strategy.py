#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian20_VolumeSpike_TrendFilter_v1"
timeframe = "6h"
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
    
    # Calculate 20-period Donchian channels on daily data
    # Upper = max(high_1d, period=20)
    # Lower = min(low_1d, period=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (wait for daily bar to close)
    donchian_upper_6h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_6h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Trend filter: 50-period EMA on daily close
    close_series = pd.Series(close_1d)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i]) or \
           np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above Donchian upper + volume spike + price > EMA50 (uptrend)
            if price > donchian_upper_6h[i] and volume_spike and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volume spike + price < EMA50 (downtrend)
            elif price < donchian_lower_6h[i] and volume_spike and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below Donchian lower (reversal signal)
            if price < donchian_lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above Donchian upper (reversal signal)
            if price > donchian_upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals