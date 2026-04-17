#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + 12h EMA34 Trend Filter
Long: Price breaks above Donchian(20) high + volume > 2x 12h volume MA + price > 12h EMA34
Short: Price breaks below Donchian(20) low + volume > 2x 12h volume MA + price < 12h EMA34
Exit: Price crosses back below/above 12h EMA34
Target: 20-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h volume MA (24-period for confirmation)
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        
        if position == 0:
            # Long: break above Donchian high + volume spike + 12h uptrend
            if price > donchian_high[i] and vol > 2.0 * vol_ma and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + volume spike + 12h downtrend
            elif price < donchian_low[i] and vol > 2.0 * vol_ma and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below 12h EMA34
            if price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h EMA34
            if price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0