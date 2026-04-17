#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h EMA trend filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 1d Donchian high, 12h EMA is rising, and volume > 1.5x 20-period average.
Short when price breaks below 1d Donchian low, 12h EMA is falling, and volume > 1.5x 20-period average.
Exit on opposite Donchian breakout or when 12h EMA flips direction.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Works in bull markets (trend continuation) and bear markets (counter-trend reversals at extremes).
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
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA (34-period) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_prev = np.roll(ema_34_12h, 1)
    ema_34_12h_prev[0] = np.nan
    ema_rising = ema_34_12h > ema_34_12h_prev
    ema_falling = ema_34_12h < ema_34_12h_prev
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Donchian high, 12h EMA rising, with volume
            if (close[i] > upper_20_aligned[i] and 
                ema_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low, 12h EMA falling, with volume
            elif (close[i] < lower_20_aligned[i] and 
                  ema_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below daily Donchian low OR 12h EMA starts falling
            if (close[i] < lower_20_aligned[i] or 
                ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above daily Donchian high OR 12h EMA starts rising
            if (close[i] > upper_20_aligned[i] or 
                ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34_1dDonchian20_Volume"
timeframe = "4h"
leverage = 1.0