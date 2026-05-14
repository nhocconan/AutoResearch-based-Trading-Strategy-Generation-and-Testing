#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above 20-bar high AND 4h volume > 1.5x 20-bar avg AND 12h EMA(34) rising.
Short when price breaks below 20-bar low AND 4h volume > 1.5x 20-bar avg AND 12h EMA(34) falling.
Exit when price touches 12-bar midpoint.
Uses 4h for execution and volume, 12h for EMA trend filter.
Designed to capture strong trends with volume confirmation across bull and bear markets.
Target: 20-50 trades/year per symbol.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_rising = ema_34_12h > np.roll(ema_34_12h, 1)
    ema_34_falling = ema_34_12h < np.roll(ema_34_12h, 1)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_falling)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_rising_aligned[i]) or 
            np.isnan(ema_34_falling_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_high = close[i] > donch_high_aligned[i]
        breakout_low = close[i] < donch_low_aligned[i]
        
        # Exit condition: touch midpoint
        touch_mid = abs(close[i] - donch_mid_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and rising EMA
            if (breakout_high and volume_confirmed and ema_34_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and falling EMA
            elif (breakout_low and volume_confirmed and ema_34_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0