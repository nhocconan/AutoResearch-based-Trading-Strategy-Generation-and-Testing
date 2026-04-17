#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above upper Donchian channel with volume > 1.5x 4h avg volume AND 12h EMA34 rising.
Short when price breaks below lower Donchian channel with volume > 1.5x 4h avg volume AND 12h EMA34 falling.
Exit when price touches the opposite Donchian channel or 12h EMA34.
Uses 4h for execution and volume, 12h for EMA trend filter.
Designed to capture strong momentum moves with volume confirmation while filtering counter-trend noise.
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
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_rising = ema_34_12h > np.roll(ema_34_12h, 1)
    ema_34_falling = ema_34_12h < np.roll(ema_34_12h, 1)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    
    # Align 4h indicators to primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Align 12h EMA to primary timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_34_rising_aligned[i]) or
            np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > donchian_high_aligned[i]
        breakout_lower = close[i] < donchian_low_aligned[i]
        
        # Exit conditions: touch opposite channel or 12h EMA34
        touch_lower_channel = close[i] <= donchian_low_aligned[i]
        touch_upper_channel = close[i] >= donchian_high_aligned[i]
        touch_ema = abs(close[i] - ema_34_12h[-1]) < 0.005 * close[i] if len(ema_34_12h) > 0 else False
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and rising 12h EMA
            if (breakout_upper and volume_confirmed and ema_34_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and falling 12h EMA
            elif (breakout_lower and volume_confirmed and ema_34_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch lower Donchian channel or 12h EMA34
            if touch_lower_channel or touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper Donchian channel or 12h EMA34
            if touch_upper_channel or touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_4hVolume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0