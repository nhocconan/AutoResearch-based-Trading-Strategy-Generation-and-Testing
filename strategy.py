#!/usr/bin/env python3
"""
1h Volume Spike + 4h Donchian Breakout + 1d EMA Trend Filter
Strategy combines volume spikes (breakout momentum) with 4h Donchian breakouts for direction,
filtered by 1d EMA50 trend. Designed for low trade frequency (15-30/year) with clear edge in
both bull and bear markets via trend filtering and volume confirmation.
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
    
    # Get 4h data for Donchian channels (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_donchian_high = price > donchian_high_aligned[i]
        below_donchian_low = price < donchian_low_aligned[i]
        above_1d_ema = price > ema_50_1d_aligned[i]
        below_1d_ema = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, above 1d EMA, volume spike
            if above_donchian_high and above_1d_ema and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low, below 1d EMA, volume spike
            elif below_donchian_low and below_1d_ema and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.20
            # Exit: price breaks below 4h Donchian low or breaks below 1d EMA
            if below_donchian_low or below_1d_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.20
            # Exit: price breaks above 4h Donchian high or breaks above 1d EMA
            if above_donchian_high or above_1d_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_4hDonchian_1dEMA50"
timeframe = "1h"
leverage = 1.0