#!/usr/bin/env python3
"""
12h_Donchian_Breakout_20_1dEMA34_Volume
Hypothesis: 12-hour Donchian channel breakout (20-period) with 1-day EMA(34) trend filter and volume confirmation captures breakouts in both bull and bear markets. The 12h timeframe reduces trade frequency to manageable levels (~20-30/year) while EMA34 filters false breakouts and volume ensures institutional participation. Designed for BTC/ETH robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel: 20-period high/low
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for Donchian (20) and volume MA (20)
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1d_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long: break above Donchian high with volume in uptrend
            if price > donchian_high and vol_ok and price > ema_trend:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low with volume in downtrend
            elif price < donchian_low and vol_ok and price < ema_trend:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Maintain long until price breaks below Donchian low or trend reverses
            if price < donchian_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Maintain short until price breaks above Donchian high or trend reverses
            if price > donchian_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian_Breakout_20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0