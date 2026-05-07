#!/usr/bin/env python3
# 12h_WilsonBreakout_1wTrend_VolumeConfirm
# Hypothesis: Uses 12h Donchian breakout (20) filtered by 1w EMA trend and volume spikes.
# Long when price breaks above Donchian(20) high + price > 1w EMA50 + volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low + price < 1w EMA50 + volume spike.
# Exit when price crosses back below/above Donchian(20) opposite band.
# Designed for 12h to target 50-150 total trades over 4 years with low turnover.
# Trend filter works in bull/bear by aligning with higher timeframe direction.

name = "12h_WilsonBreakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + above 1w EMA50 + volume spike
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below 1w EMA50 + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals