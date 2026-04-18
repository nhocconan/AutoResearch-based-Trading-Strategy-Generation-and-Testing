#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and Daily EMA Trend Filter
Uses Donchian channel breakout (20-period) from 1d timeframe combined with volume confirmation
and daily EMA trend filter. Designed for low trade frequency with strong edge in both bull
and bear markets by taking breakouts in direction of higher timeframe trend. Focuses on
Donchian breakouts to capture strong momentum moves while avoiding noise.
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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channel (20-period)
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    lookback = 20
    upper_band = np.full(len(high_1d), np.nan)
    lower_band = np.full(len(low_1d), np.nan)
    
    for i in range(lookback, len(high_1d)):
        upper_band[i] = np.max(high_1d[i-lookback:i])
        lower_band[i] = np.min(low_1d[i-lookback:i])
    
    # Align Donchian levels to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get 1d data for EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_level = upper_band_aligned[i]
        lower_level = lower_band_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above daily EMA
            if (price > upper_level and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume spike and below daily EMA
            elif (price < lower_level and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions: reverse signal (break below lower band)
            if price < lower_level:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions: reverse signal (break above upper band)
            if price > upper_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_Volume_DailyEMA34"
timeframe = "12h"
leverage = 1.0