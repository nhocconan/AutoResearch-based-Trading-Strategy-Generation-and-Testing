#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Spike
Long: Price breaks above Donchian(20) high + price > 12h EMA34 + volume > 2x 12h volume MA
Short: Price breaks below Donchian(20) low + price < 12h EMA34 + volume > 2x 12h volume MA
Exit: Opposite Donchian break or price crosses 12h EMA34
Target: 25-35 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h volume MA (24-period for spike detection)
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean().values
    volume_ma_24_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_24)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_24_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + trend + volume spike
            if price > high_max[i] and price > ema_34_12h_aligned[i] and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + trend + volume spike
            elif price < low_min[i] and price < ema_34_12h_aligned[i] and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below Donchian low OR price crosses below 12h EMA34
            if price < low_min[i] or price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR price crosses above 12h EMA34
            if price > high_max[i] or price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0