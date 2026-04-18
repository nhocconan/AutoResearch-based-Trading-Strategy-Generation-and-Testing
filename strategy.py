#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and 1d EMA Trend Filter
Based on successful patterns: Donchian breakouts work well on 12h timeframe with volume confirmation.
Uses 1d EMA for trend filter to align with higher timeframe direction. Designed for low trade frequency
(12-37 trades/year target) with strong edge in both bull and bear markets by taking breakouts in direction
of higher timeframe trend. Focuses on minimizing false breakouts with volume confirmation and trend alignment.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # We need to get 12h high/low for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian upper and lower bands
    high_ma = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (no additional delay needed for Donchian)
    donchian_upper = align_htf_to_ltf(prices, df_12h, high_ma)
    donchian_lower = align_htf_to_ltf(prices, df_12h, low_ma)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and above 1d EMA
            if (price > upper and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume spike and below 1d EMA
            elif (price < lower and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
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

name = "12h_Donchian_Breakout_Volume_Spike_1dEMA34"
timeframe = "12h"
leverage = 1.0