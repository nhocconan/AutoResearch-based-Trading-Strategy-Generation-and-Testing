#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Long: price breaks above Donchian upper (20-period 12h high) + price > 1d EMA50 + volume > 1.5x 20-period avg volume
- Short: price breaks below Donchian lower (20-period 12h low) + price < 1d EMA50 + volume > 1.5x 20-period avg volume
- Exit: price closes back inside the Donchian channel (mean reversion in choppy markets)
- Uses 12h timeframe to reduce trade frequency and fee drag
- Donchian breakout captures momentum; EMA50 filters for trend alignment; volume confirms conviction
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
- Works in both bull (trend continuation) and bear (mean reversion after extreme moves) regimes
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
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_upper = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 20, 50)  # Need 20 for Donchian, 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > donchian_upper_aligned[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower_aligned[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA50 + volume spike
            if breakout_up and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + price < 1d EMA50 + volume spike
            elif breakout_down and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes back inside Donchian channel (mean reversion)
            if close[i] < donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes back inside Donchian channel (mean reversion)
            if close[i] > donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeSpike_MeanReversionExit"
timeframe = "12h"
leverage = 1.0