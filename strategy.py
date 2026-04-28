#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (1w Camarilla) and volume confirmation.
# Uses 6h primary timeframe to balance trade frequency and signal quality.
# Donchian breakouts provide clear structure, filtered by weekly Camarilla pivot bias (long above H4, short below L4)
# and volume spikes to avoid false breakouts. Works in both bull and bear markets by
# following the weekly pivot structure while using Donchian channels for entry timing.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Donchian20_1wCamarilla_H4L4_Bias_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for weekly Camarilla calculation (need 5 days for 1 week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla levels (using prior week's range)
    # H4 = close + 1.1*(high - low)/2
    # L4 = close - 1.1*(high - low)/2
    weekly_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * weekly_range / 2
    camarilla_l4 = close_1d - 1.1 * weekly_range / 2
    
    # Align weekly Camarilla to 6h timeframe (wait for weekly close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 6h Donchian(20) channels
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Donchian needs 20, volume MA needs 20, plus buffer for safety
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(high_ma_20[i]) or
            np.isnan(low_ma_20[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Weekly Camarilla bias: long above H4, short below L4
        bias_long = close[i] > camarilla_h4_aligned[i]
        bias_short = close[i] < camarilla_l4_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_ma_20[i]
        short_breakout = close[i] < low_ma_20[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = bias_long and long_breakout and vol_confirm
        short_entry = bias_short and short_breakout and vol_confirm
        
        # Exit conditions: opposite Donchian breakout
        long_exit = close[i] < low_ma_20[i]
        short_exit = close[i] > high_ma_20[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals