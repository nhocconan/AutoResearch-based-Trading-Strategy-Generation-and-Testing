#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and 1d EMA Trend Filter
Trades breakouts of 20-period Donchian channels on 12h timeframe, filtered by
1d EMA trend direction and volume confirmation. Designed for low trade frequency
(12-37 trades/year) with strong edge in both bull and bear markets by taking
breakouts only in direction of higher timeframe trend. Uses discrete position
sizing to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Donchian channel (20-period) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Volume spike detection (2x 4-period average on 12h)
    df_12h_vol = get_htf_data(prices, '12h')
    vol_12h = df_12h_vol['volume'].values
    vol_ma = pd.Series(vol_12h).rolling(window=4, min_periods=4).mean().values
    volume_spike = vol_12h > (2.0 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h_vol, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and above 1d EMA
            if (price > upper and vol_spike and price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and below 1d EMA
            elif (price < lower and vol_spike and price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until reversal signal
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian
            if price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal signal
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian
            if price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Spike_1dEMA34"
timeframe = "12h"
leverage = 1.0