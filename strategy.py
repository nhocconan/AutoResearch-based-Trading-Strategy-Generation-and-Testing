#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h Trend + Volume Spike
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and 12h EMA34 up.
Short when price breaks below Donchian(20) low with volume > 1.5x average and 12h EMA34 down.
Exit when price crosses the opposite Donchian band or volume drops below average.
Designed for 4h to capture trends with moderate trade frequency (~20-40/year) and avoid whipsaws.
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and bullish 12h trend
            if price > donchian_high[i] and vol > 1.5 * vol_ma and ema_34_4h[i] > ema_34_4h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and bearish 12h trend
            elif price < donchian_low[i] and vol > 1.5 * vol_ma and ema_34_4h[i] < ema_34_4h[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR volume drops below average
            if price < donchian_low[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR volume drops below average
            if price > donchian_high[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0