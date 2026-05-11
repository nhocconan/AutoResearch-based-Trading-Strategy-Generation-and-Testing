#!/usr/bin/env python3
name = "6h_Alligator_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly trend: close above/below weekly EMA21
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    weekly_up = close > ema_1w_aligned
    
    # Williams Alligator: 3 SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period SMA, 8 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values    # 8-period SMA, 5 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values     # 5-period SMA, 3 bars ahead
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > 1.8 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Alligator (13+8 shift)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + weekly up + volume filter
            if lips[i] > teeth[i] > jaw[i] and weekly_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + weekly down + volume filter
            elif jaw[i] > teeth[i] > lips[i] and not weekly_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or weekly trend down
            if not (lips[i] > teeth[i] > jaw[i]) or not weekly_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or weekly trend up
            if not (jaw[i] > teeth[i] > lips[i]) or weekly_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals