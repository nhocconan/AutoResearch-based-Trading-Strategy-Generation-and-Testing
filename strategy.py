#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w trend filter + volume confirmation.
Long when price breaks above 20-day high AND 1w close > 1w EMA34 AND volume > 1.5x 20-day avg volume.
Short when price breaks below 20-day low AND 1w close < 1w EMA34 AND volume > 1.5x 20-day avg volume.
Exit on opposite Donchian breakout or volume drop below avg volume.
Uses 1d for price/volume, 1w for trend filter.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to lower timeframe (prices index)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    # Align 1w indicators
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        avg_vol_val = avg_volume_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = vol > 1.5 * avg_vol_val
        
        # Trend filter: 1w close > 1w EMA34 for long bias, < for short bias
        trend_long = close_1w[i] > ema34_1w_val  # Note: close_1w[i] is current 1w close (may be forming)
        trend_short = close_1w[i] < ema34_1w_val
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirm + trend long
            if price > donchian_high_val and volume_confirm and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirm + trend short
            elif price < donchian_low_val and volume_confirm and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume drops below average
            if price < donchian_low_val or vol < avg_vol_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume drops below average
            if price > donchian_high_val or vol < avg_vol_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Confirm"
timeframe = "1d"
leverage = 1.0