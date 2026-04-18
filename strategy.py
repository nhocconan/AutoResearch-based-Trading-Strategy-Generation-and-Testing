# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_Volume_Trend_v1
Hypothesis: Price breaking Donchian(20) high/low with volume spike and 12h EMA34 trend captures breakouts in both bull and bear markets. Donchian channels capture volatility expansion, volume confirms institutional interest, and EMA34 filter ensures trend alignment. Designed for low trade frequency (<25/year) to minimize fee drag while catching explosive moves.
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
    
    # Donchian Channels (20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max()
    lower_donchian = low_series.rolling(window=20, min_periods=20).min()
    upper = upper_donchian.values
    lower = lower_donchian.values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(upper[i]) or
            np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_spike = volume_spike[i]
        upper_val = upper[i]
        lower_val = lower[i]
        
        if position == 0:
            # Long: price > upper Donchian with volume spike and above 12h EMA34
            if price > upper_val and vol_spike and price > ema_12h_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian with volume spike and below 12h EMA34
            elif price < lower_val and vol_spike and price < ema_12h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < lower Donchian or below 12h EMA34
            if price < lower_val or price < ema_12h_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > upper Donchian or above 12h EMA34
            if price > upper_val or price > ema_12h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0