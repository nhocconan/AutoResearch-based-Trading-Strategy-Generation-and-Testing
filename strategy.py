#!/usr/bin/env python3
"""
Hypothesis: 4h 1-day Bollinger Band squeeze breakout with volume confirmation and trend filter.
Uses 1-day Bollinger Band width percentile to detect low volatility squeeze (BBW < 20th percentile),
1-day Bollinger Band breakout for entry direction, and 4-hour volume > 1.5x 20-period average for confirmation.
Long when price breaks above upper BB in squeeze with volume confirmation.
Short when price breaks below lower BB in squeeze with volume confirmation.
Exit when price returns to middle Bollinger Band.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Works in both bull and bear markets by trading volatility breakouts regardless of direction.
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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1-day Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate Bollinger Band Width
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate 20th percentile of BB width for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Align BB levels and squeeze to 4h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or 
            np.isnan(bb_squeeze_aligned[i]) or
            i - 50 < 20):  # Ensure 4h volume MA is ready
            signals[i] = 0.0
            continue
        
        # Entry conditions: BB breakout + squeeze + volume spike
        breakout_long = close[i] > bb_upper_aligned[i]
        breakout_short = close[i] < bb_lower_aligned[i]
        in_squeeze = bb_squeeze_aligned[i] > 0.5
        vol_confirm = volume[i] > vol_ma_20[min(i, len(vol_ma_20)-1)] * 1.5 if i < len(vol_ma_20) else False
        
        long_entry = breakout_long and in_squeeze and vol_confirm
        short_entry = breakout_short and in_squeeze and vol_confirm
        
        # Exit when price returns to middle Bollinger Band
        exit_long = position == 1 and close[i] < bb_middle_aligned[i]
        exit_short = position == -1 and close[i] > bb_middle_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_bb_squeeze_breakout"
timeframe = "4h"
leverage = 1.0