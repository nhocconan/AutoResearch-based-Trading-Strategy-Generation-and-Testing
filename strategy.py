#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d EMA Trend Filter
Designed for low trade frequency (~25-40 trades/year) with strong performance
in both bull and bear markets by combining price channel breakouts with
volume confirmation and higher timeframe trend alignment.
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
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.8x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_max[i]
        lower_channel = low_min[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation and above 1d EMA
            if (price > upper_channel and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume confirmation and below 1d EMA
            elif (price < lower_channel and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian band (contrarian exit)
            if price < lower_channel:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian band (contrarian exit)
            if price > upper_channel:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_1dEMA34"
timeframe = "4h"
leverage = 1.0