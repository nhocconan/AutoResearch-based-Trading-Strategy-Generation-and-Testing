#!/usr/bin/env python3
"""
12h_1w_Donchian_20_Breakout_Volume_Trend
Hypothesis: Trade breakouts above/below weekly Donchian channels (20-period) in direction of weekly EMA(34) trend, confirmed by volume >1.5x 24-period average. Position size 0.25 targeting ~20 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
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
    
    # Get 1w data for Donchian and EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high/low for Donchian (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    donchian_period = 20
    if len(high_1w) >= donchian_period:
        for i in range(donchian_period - 1, len(high_1w)):
            donchian_high[i] = np.max(high_1w[i - donchian_period + 1:i + 1])
            donchian_low[i] = np.min(low_1w[i - donchian_period + 1:i + 1])
    
    # Weekly EMA trend filter (34-period)
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Align weekly indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, donchian_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and above weekly EMA
            if close[i] > donchian_high_aligned[i] and vol_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and below weekly EMA
            elif close[i] < donchian_low_aligned[i] and vol_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below weekly Donchian low or below weekly EMA
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above weekly Donchian high or above weekly EMA
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0