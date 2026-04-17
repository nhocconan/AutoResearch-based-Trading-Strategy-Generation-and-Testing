#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Supertrend(10,3) filter + 6h Donchian(20) breakout + volume confirmation.
Long when price breaks above 6h Donchian(20) high with 12h Supertrend uptrend and volume > 1.5x 20-period volume average.
Short when price breaks below 6h Donchian(20) low with 12h Supertrend downtrend and volume > 1.5x 20-period volume average.
Supertrend on 12h timeframe provides strong trend filter to avoid whipsaws in ranging markets, while Donchian breakout captures momentum.
Designed to work in bull markets (breakout with uptrend) and bear markets (breakdown with downtrend).
Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300 total.
Size: 0.25-0.30.
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
    
    # Calculate 12h Supertrend(10,3)
    def supertrend(high_vals, low_vals, close_vals, atr_period, multiplier):
        # True Range
        tr1 = pd.DataFrame(high_vals - low_vals)
        tr2 = pd.DataFrame(np.abs(high_vals - np.roll(close_vals, 1)))
        tr3 = pd.DataFrame(np.abs(low_vals - np.roll(close_vals, 1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        tr[0] = high_vals[0] - low_vals[0]  # first period
        
        # ATR
        atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
        
        # Basic Upper and Lower Bands
        hl_avg = (high_vals + low_vals) / 2
        upper_basic = hl_avg + multiplier * atr
        lower_basic = hl_avg - multiplier * atr
        
        # Final Upper and Lower Bands
        upper = np.full_like(close_vals, np.nan)
        lower = np.full_like(close_vals, np.nan)
        
        for i in range(atr_period, len(close_vals)):
            # Upper Band
            if upper_basic[i] < upper[i-1] or close_vals[i-1] > upper[i-1]:
                upper[i] = upper_basic[i]
            else:
                upper[i] = upper[i-1]
            
            # Lower Band
            if lower_basic[i] > lower[i-1] or close_vals[i-1] < lower[i-1]:
                lower[i] = lower_basic[i]
            else:
                lower[i] = lower[i-1]
        
        # Supertrend
        supertrend_val = np.full_like(close_vals, np.nan)
        direction = np.ones_like(close_vals, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        for i in range(atr_period, len(close_vals)):
            if close_vals[i] > upper[i-1]:
                direction[i] = 1
            elif close_vals[i] < lower[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            if direction[i] == 1:
                supertrend_val[i] = lower[i]
            else:
                supertrend_val[i] = upper[i]
        
        return supertrend_val, direction
    
    st_12h, st_dir_12h = supertrend(high_12h, low_12h, close_12h, 10, 3)
    
    # Calculate 6h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 6h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Supertrend direction to 6h timeframe
    st_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, st_dir_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(st_dir_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 6h Donchian(20) high with 12h Supertrend uptrend and volume
            if (close[i] > donchian_upper[i] and 
                st_dir_12h_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian(20) low with 12h Supertrend downtrend and volume
            elif (close[i] < donchian_lower[i] and 
                  st_dir_12h_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 6h Donchian(20) low (opposite side of channel)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 6h Donchian(20) high (opposite side of channel)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hSupertrend10_3_Donchian20_Breakout_Volume_Confirm"
timeframe = "6h"
leverage = 1.0