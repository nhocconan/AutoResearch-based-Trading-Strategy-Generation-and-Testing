#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h Supertrend trend filter and volume spike confirmation
# Long when price breaks above upper Donchian(20) AND 12h Supertrend = bullish AND volume > 2.0x 20-bar avg
# Short when price breaks below lower Donchian(20) AND 12h Supertrend = bearish AND volume > 2.0x 20-bar avg
# Exit when price crosses 12h Supertrend line (trend change)
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining exposure.
# Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years) to avoid overtrading.
# Donchian channels provide objective breakout levels, Supertrend filters false breakouts in chop,
# Volume spike confirms institutional participation. Works in bull/bear via trend filter.

name = "4h_Donchian20_Supertrend12h_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, multiplier=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_12h[0] - low_12h[0]  # First period TR
    
    # ATR calculation
    atr_12h = tr.rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + 3.0 * atr_12h
    lower_band = hl2 - 3.0 * atr_12h
    
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[9] = upper_band[9]  # Initialize
    direction[9] = 1
    
    for i in range(10, len(close_12h)):
        if close_12h[i-1] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 20-bar average volume (tight to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_supertrend = supertrend_aligned[i]
        curr_direction = direction_aligned[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Supertrend (trend change to bearish)
            if curr_close < curr_supertrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Supertrend (trend change to bullish)
            if curr_close > curr_supertrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND Supertrend bullish AND volume confirmation
            if curr_close > curr_dc_upper and curr_direction == 1 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND Supertrend bearish AND volume confirmation
            elif curr_close < curr_dc_lower and curr_direction == -1 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals