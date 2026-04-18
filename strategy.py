#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1d_EMA34_Volume_Confirmation
Hypothesis: Buy when price breaks above 20-period Donchian upper band on 12h timeframe with volume spike and above 1d EMA34 trend; sell when price breaks below lower band with volume spike and below 1d EMA34. Donchian channels capture volatility expansion and breakouts, volume confirms institutional participation, and 1d EMA34 ensures alignment with long-term trend. Designed for very low trade frequency (<20/year) to minimize fee drag while capturing major trends in both bull and bear markets.
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
    
    # 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max()
    lower_donchian = low_series.rolling(window=20, min_periods=20).min()
    upper_donchian_arr = upper_donchian.values
    lower_donchian_arr = lower_donchian.values
    
    # Volume spike: >1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.8 * vol_ma.values)
    
    # 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(upper_donchian_arr[i]) or
            np.isnan(lower_donchian_arr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_1d_val = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        upper = upper_donchian_arr[i]
        lower = lower_donchian_arr[i]
        
        if position == 0:
            # Long: price > upper Donchian with volume spike and above 1d EMA34
            if price > upper and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian with volume spike and below 1d EMA34
            elif price < lower and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < lower Donchian or below 1d EMA34
            if price < lower or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > upper Donchian or above 1d EMA34
            if price > upper or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_1d_EMA34_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0