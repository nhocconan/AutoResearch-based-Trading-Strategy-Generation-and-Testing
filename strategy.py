#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above Donchian upper channel with volume > 1.8x 4h avg volume AND 1d EMA50 rising.
Short when price breaks below Donchian lower channel with volume > 1.8x 4h avg volume AND 1d EMA50 falling.
Exit when price touches the 1d EMA50.
Uses 4h for execution and volume, 1d for EMA trend filter and Donchian calculation.
Designed to work in both bull and bear markets by following the 1d trend with volume confirmation.
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
    
    # Get 1d data for EMA trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Align 1d EMA and Donchian to primary timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Calculate Donchian channels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: 20-period high
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period low
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h volume MA to primary timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x 20-bar average
        volume_confirmed = volume[i] > 1.8 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > donchian_upper_aligned[i]
        breakout_lower = close[i] < donchian_lower_aligned[i]
        
        # Exit condition: touch 1d EMA50
        touch_ema = abs(close[i] - ema_50_aligned[i]) < 0.003 * close[i]  # within 0.3%
        
        if position == 0:
            # Long: break above upper channel with volume confirmation and rising 1d EMA
            if (breakout_upper and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume confirmation and falling 1d EMA
            elif (breakout_lower and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch 1d EMA50
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch 1d EMA50
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0