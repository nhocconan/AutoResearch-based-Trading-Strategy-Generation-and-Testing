#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Supertrend trend filter and Donchian breakout.
# Long: Price breaks above Donchian(20) upper band + 12h Supertrend = bullish + volume > 1.5x average.
# Short: Price breaks below Donchian(20) lower band + 12h Supertrend = bearish + volume > 1.5x average.
# Uses Donchian channels for breakouts, Supertrend for trend filtering, volume for confirmation.
# Time filter: 00-23 UTC (all hours).
# Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_up = np.full(n, np.nan)
    donchian_down = np.full(n, np.nan)
    for i in range(20, n):
        donchian_up[i] = np.max(high[i-20:i])
        donchian_down[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 12h
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr_12h = np.concatenate([[np.nan], tr_12h])
    
    atr_12h = np.full(len(close_12h), np.nan)
    for i in range(atr_period, len(close_12h)):
        atr_12h[i] = np.nanmean(tr_12h[i-atr_period+1:i+1])
    
    # Supertrend calculation
    supertrend_12h = np.full(len(close_12h), np.nan)
    direction_12h = np.full(len(close_12h), np.nan)  # 1 for up, -1 for down
    
    # Initialize
    if not np.isnan(atr_12h[atr_period]):
        hl2_12h = (high_12h[atr_period] + low_12h[atr_period]) / 2
        upper_band_12h = hl2_12h + multiplier * atr_12h[atr_period]
        lower_band_12h = hl2_12h - multiplier * atr_12h[atr_period]
        supertrend_12h[atr_period] = upper_band_12h
        direction_12h[atr_period] = -1  # start with down
    
    for i in range(atr_period + 1, len(close_12h)):
        hl2 = (high_12h[i] + low_12h[i]) / 2
        
        upper_band = hl2 + multiplier * atr_12h[i]
        lower_band = hl2 - multiplier * atr_12h[i]
        
        if i == atr_period + 1:
            prev_supertrend = supertrend_12h[i-1]
            prev_direction = direction_12h[i-1]
        else:
            prev_supertrend = supertrend_12h[i-1]
            prev_direction = direction_12h[i-1]
        
        if not np.isnan(prev_supertrend):
            if prev_direction == 1:  # was uptrend
                supertrend_12h[i] = max(lower_band, prev_supertrend)
                if close_12h[i] > supertrend_12h[i]:
                    direction_12h[i] = 1
                else:
                    direction_12h[i] = -1
                    supertrend_12h[i] = upper_band
            else:  # was downtrend
                supertrend_12h[i] = min(upper_band, prev_supertrend)
                if close_12h[i] < supertrend_12h[i]:
                    direction_12h[i] = -1
                else:
                    direction_12h[i] = 1
                    supertrend_12h[i] = lower_band
        else:
            supertrend_12h[i] = np.nan
            direction_12h[i] = np.nan
    
    # Align 12h Supertrend direction to 4h
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_down[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(direction_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        trend = direction_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian up + bullish trend + volume
            if (price > donchian_up[i] and trend == 1 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian down + bearish trend + volume
            elif (price < donchian_down[i] and trend == -1 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian down
            if price < donchian_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian up
            if price > donchian_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Supertrend_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0