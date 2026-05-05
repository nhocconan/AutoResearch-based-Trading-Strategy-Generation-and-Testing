#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation (2x 20-period MA)
# Long when: price breaks above Donchian(20) upper band AND price > 1d EMA50 (uptrend) AND volume > 2x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND price < 1d EMA50 (downtrend) AND volume > 2x 20-period MA
# Exit when: price reverses to Donchian(20) midpoint OR trend filter reverses
# Uses Donchian channels for structure, 1d EMA for trend filter, volume spike for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Breakout_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) channels on 12h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high
        donchian_lower = lowest_low
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > donchian_upper) & (np.roll(close, 1) <= np.roll(donchian_upper, 1))
    donchian_breakout_down = (close < donchian_lower) & (np.roll(close, 1) >= np.roll(donchian_lower, 1))
    donchian_revert_up = (close > donchian_mid) & (np.roll(close, 1) <= np.roll(donchian_mid, 1))
    donchian_revert_down = (close < donchian_mid) & (np.roll(close, 1) >= np.roll(donchian_mid, 1))
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_rising = np.diff(ema_50_1d, prepend=np.nan) > 0
        ema_falling = np.diff(ema_50_1d, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_1d), False)
        ema_falling = np.full(len(close_1d), False)
    
    # Align 1d EMA trend to 12h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + 1d EMA rising + volume filter
            if (donchian_breakout_up[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + 1d EMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian revert to mid OR 1d EMA turns falling
            if (donchian_revert_down[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian revert to mid OR 1d EMA turns rising
            if (donchian_revert_up[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals