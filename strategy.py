#!/usr/bin/env python3
"""
6h Donchian Breakout with Daily Trend Filter and Volume Confirmation
Breakout of 20-period Donchian channel on 6h, filtered by 1d EMA trend and volume spike.
Designed for low trade frequency with clear edge in both bull and bear markets by
trading breakouts in direction of higher timeframe trend. Focuses on strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Donchian channel (20-period)
    donchian_len = 20
    # Upper band: highest high of last donchian_len periods
    upper_band = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    # Lower band: lowest low of last donchian_len periods
    lower_band = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = max(100, donchian_len)  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above 1d EMA
            if (price > upper and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume spike and below 1d EMA
            elif (price < lower and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: reverse signal (break below lower band)
            if price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: reverse signal (break above upper band)
            if price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0