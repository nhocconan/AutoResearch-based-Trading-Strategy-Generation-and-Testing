#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1D Trend Filter
Long: Price breaks above 20-period Donchian high + volume > 1.5x 4h volume MA + price > 1D EMA50
Short: Price breaks below 20-period Donchian low + volume > 1.5x 4h volume MA + price < 1D EMA50
Exit: Opposite Donchian break or 1D EMA cross
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume moving average (20-period for confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1D EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: Donchian breakout + volume + 1D trend
            if price > donchian_high[i] and vol > 1.5 * vol_ma_20[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: Donchian breakdown + volume + 1D trend
            elif price < donchian_low[i] and vol > 1.5 * vol_ma_20[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
                
        elif position == 1:
            # Long exit: Donchian breakdown OR price below 1D EMA50
            if price < donchian_low[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:
            # Short exit: Donchian breakout OR price above 1D EMA50
            if price > donchian_high[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_1DEMA50"
timeframe = "4h"
leverage = 1.0