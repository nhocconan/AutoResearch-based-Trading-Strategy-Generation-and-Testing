#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Enter long when price breaks above upper Donchian channel and 1d EMA is rising.
# Enter short when price breaks below lower Donchian channel and 1d EMA is falling.
# Exit when price reverses to the middle of the Donchian channel or trend changes.
# Volume confirmation: current volume > 1.5x 20-period volume EMA.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 20-period volume EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Align 1d indicators to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(middle_channel[i]) or np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above upper channel in uptrend with volume
            if (close[i] > upper_channel[i] and
                trend_up_aligned[i] > 0.5 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower channel in downtrend with volume
            elif (close[i] < lower_channel[i] and
                  trend_up_aligned[i] <= 0.5 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle channel or trend turns down
            if (close[i] < middle_channel[i] or
                trend_up_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle channel or trend turns up
            if (close[i] > middle_channel[i] or
                trend_up_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals