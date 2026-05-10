#!/usr/bin/env python3
# 4h_Breakout_20Period_1dTrend_VolumeSpike
# Hypothesis: Price breaking 20-period high/low on 4h chart with 1-day trend filter and volume spike.
# The 20-period high/low acts as a dynamic support/resistance level. Breakouts with volume and daily trend
# filter capture sustained moves while avoiding counter-trend trades. Volume spike confirms institutional
# participation. Designed for moderate trade frequency to balance signal quality and cost.
# Targets 20-40 trades per year on 4h timeframe.

name = "4h_Breakout_20Period_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period high and low on 4h chart
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_20 = high_series.rolling(window=20, min_periods=20).max().values
    low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA (34), 20-period high/low (20), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_long = close[i] > high_20[i]
        breakout_short = close[i] < low_20[i]
        
        if position == 0:
            # Long entry: price breaks above 20-period high + daily uptrend + volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low + daily downtrend + volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below 20-period low or daily trend turns down
            if close[i] < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above 20-period high or daily trend turns up
            if close[i] > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals