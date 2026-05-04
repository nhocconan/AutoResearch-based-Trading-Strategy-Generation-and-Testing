#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period high) AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian lower (20-period low) AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Uses 4h for primary timeframe to control trade frequency, 1d for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 25-50 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    # We need at least 20 periods for Donchian calculation
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1d uptrend AND volume spike
            if (close[i] > high_rolling_max[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1d downtrend AND volume spike
            elif (close[i] < low_rolling_min[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR 1d trend changes to downtrend
            if (close[i] < low_rolling_min[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR 1d trend changes to uptrend
            if (close[i] > high_rolling_max[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals