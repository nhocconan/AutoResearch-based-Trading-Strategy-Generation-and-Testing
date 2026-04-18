#!/usr/bin/env python3
"""
6h_12h_Donchian20_Breakout_Volume_TrendFilter
Hypothesis: Trade Donchian(20) breakouts on 6h in direction of 12h EMA(34) trend, confirmed by volume >2x 24-period average. Uses 12h trend filter to avoid counter-trend trades and volume spike to ensure conviction. Targets 60-120 trades over 4 years (15-30/year) with position size 0.25 to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment.
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
    
    # Get 12h data for Donchian and EMA calculations
    df_12h = get_htf_data(prices, '12h')
    
    # 12h calculations (previous bar's OHLC for Donchian)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC (completed bar)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    # 12h Donchian channels (20-period) based on previous bar
    donch_high = np.full_like(high_12h, np.nan)
    donch_low = np.full_like(low_12h, np.nan)
    
    lookback = 20
    for i in range(lookback, len(high_12h)):
        donch_high[i] = np.max(prev_high_12h[i-lookback:i])
        donch_low[i] = np.min(prev_low_12h[i-lookback:i])
    
    # 12h EMA trend filter (34-period)
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h Donchian and EMA to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 2x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and above 12h EMA
            if close[i] > donch_high_aligned[i] and vol_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and below 12h EMA
            elif close[i] < donch_low_aligned[i] and vol_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or below 12h EMA
            if close[i] < donch_low_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or above 12h EMA
            if close[i] > donch_high_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0