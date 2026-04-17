#!/usr/bin/env python3
"""
1d Donchian(20) breakout + 1w EMA(34) trend + volume spike + ATR stop
Long: price > Donchian(20 high) + close > 1w EMA34 + volume > 1.5x 1d volume SMA(20)
Short: price < Donchian(20 low) + close < 1w EMA34 + volume > 1.5x 1d volume SMA(20)
Exit: Opposite breakout or price crosses Donchian midpoint
Volume spike confirms breakout strength, weekly EMA filters trend direction.
Designed for 1d timeframe to work in both bull and bear markets with selective entries.
Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 1d volume SMA(20)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        donchian_mid_val = donchian_mid[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + above weekly EMA + volume spike
            if price > donchian_high and price > ema_34_1w_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + below weekly EMA + volume spike
            elif price < donchian_low and price < ema_34_1w_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian midpoint or weekly EMA
            if price < donchian_mid_val or price < ema_34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian midpoint or weekly EMA
            if price > donchian_mid_val or price > ema_34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0