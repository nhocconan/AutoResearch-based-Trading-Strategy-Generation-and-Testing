#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_Donchian20_VolumeTrend_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian(20) for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels using previous 20 days
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Previous day's levels (shift by 1)
    prev_high_max_20 = np.concatenate([[np.nan], high_max_20[:-1]])
    prev_low_min_20 = np.concatenate([[np.nan], low_min_20[:-1]])
    
    # Align to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, prev_high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, prev_low_min_20)
    
    # Volume filter: current volume > 2.0x 20-period average (20 * 4h = ~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trend = ema_34_1w_aligned[i]
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            if price > donchian_high[i] and volume_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif price < donchian_low[i] and volume_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian low (mean reversion)
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian high (mean reversion)
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals