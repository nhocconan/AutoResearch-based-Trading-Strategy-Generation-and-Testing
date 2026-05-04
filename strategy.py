#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 20-period high AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below 20-period low AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Uses 12h for primary timeframe to minimize fee drag, 1d for trend direction and volume filter.
# Discrete sizing (0.25) to balance profit potential and risk management.
# Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Donchian20_1dTrend_VolumeConfirm"
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
    
    # Get 1d data for trend filter and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Calculate 1d volume EMA20 for volume confirmation
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ema_20_1d * 1.5)
    
    # Align 1d indicators to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate on 12h data directly
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high AND 1d uptrend AND volume spike
            if (close[i] > high_roll_max[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-period low AND 1d downtrend AND volume spike
            elif (close[i] < low_roll_min[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low OR 1d trend changes to downtrend
            if (close[i] < low_roll_min[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-period high OR 1d trend changes to uptrend
            if (close[i] > high_roll_max[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals