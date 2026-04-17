#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume spike confirmation and 1w EMA34 trend filter.
Long when price breaks above Donchian upper channel with volume > 2.0x 1d average volume AND 1w EMA34 rising.
Short when price breaks below Donchian lower channel with volume > 2.0x 1d average volume AND 1w EMA34 falling.
Exit when price touches the opposite Donchian channel or when 1w EMA34 flips direction.
Uses 1d for volume confirmation (avoids 4h noise), 1w for major trend filter, 4h for entry timing.
Designed to capture strong trends with volume confirmation while avoiding choppy markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_rising = ema_34_1w > np.roll(ema_34_1w, 1)
    ema_34_falling = ema_34_1w < np.roll(ema_34_1w, 1)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    # Calculate Donchian channels on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian(20): highest high and lowest low of last 20 periods
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or
            np.isnan(ema_34_falling_aligned[i]) or
            np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Donchian levels
        upper_channel = highest_high_aligned[i]
        lower_channel = lowest_low_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_channel
        breakout_lower = close[i] < lower_channel
        
        if position == 0:
            # Long: break above upper channel with volume confirmation and rising 1w EMA
            if (breakout_upper and volume_confirmed and ema_34_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume confirmation and falling 1w EMA
            elif (breakout_lower and volume_confirmed and ema_34_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower channel OR 1w EMA flips to falling
            if (close[i] <= lower_channel or ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper channel OR 1w EMA flips to rising
            if (close[i] >= upper_channel or ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_1wEMA34_Trend"
timeframe = "4h"
leverage = 1.0