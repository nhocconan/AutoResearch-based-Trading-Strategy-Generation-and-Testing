#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Supertrend filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-period high with 12h Supertrend uptrend and volume > 1.3x 20-period volume average.
Short when price breaks below 20-period low with 12h Supertrend downtrend and volume > 1.3x 20-period volume average.
Supertrend on 12h timeframe provides robust trend filter to avoid counter-trend breakouts in choppy markets.
Designed to work in both bull and bear markets by only taking breakouts in the direction of the higher timeframe trend.
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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    def supertrend(high_vals, low_vals, close_vals, atr_period, multiplier):
        # True Range
        tr1 = pd.Series(high_vals - low_vals)
        tr2 = pd.Series(np.abs(high_vals - np.roll(close_vals, 1)))
        tr3 = pd.Series(np.abs(low_vals - np.roll(close_vals, 1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
        
        # Basic Upper and Lower Bands
        hl_avg = (high_vals + low_vals) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = np.full_like(close_vals, np.nan, dtype=float)
        direction = np.full_like(close_vals, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        # Start from atr_period to ensure valid ATR values
        for i in range(atr_period, len(close_vals)):
            # Upper Band
            if upper_band[i] < upper_band[i-1] or close_vals[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower Band
            if lower_band[i] > lower_band[i-1] or close_vals[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Supertrend and Direction
            if supertrend[i-1] == upper_band[i-1]:
                if close_vals[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            else:
                if close_vals[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
        
        # For periods before atr_period, set direction based on simple trend
        for i in range(atr_period):
            if i == 0:
                direction[i] = 1
            else:
                direction[i] = 1 if close_vals[i] > close_vals[i-1] else -1
        
        return direction  # Return 1 for uptrend, -1 for downtrend
    
    # Calculate 12h Supertrend direction
    supertrend_12h = supertrend(high_12h, low_12h, close_12h, 10, 3.0)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Supertrend to 4h timeframe
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with 12h uptrend and volume
            if (close[i] > donchian_upper[i] and 
                supertrend_12h_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with 12h downtrend and volume
            elif (close[i] < donchian_lower[i] and 
                  supertrend_12h_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (opposite side of channel)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (opposite side of channel)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hSupertrend10_3.0_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0