#!/usr/bin/env python3
# 12h_Donchian_20_1dTrend_VolumeSpike
# Hypothesis: 12-hour Donchian(20) breakout in the direction of 1-day EMA34 trend with volume spike confirmation.
# Enters long when price breaks above 20-period Donchian high and 1-day trend is up (close > EMA34).
# Enters short when price breaks below 20-period Donchian low and 1-day trend is down (close < EMA34).
# Uses volume spike (volume > 1.5x 20-period average) to confirm breakout strength.
# Designed to capture sustained moves in both bull and bear markets by aligning with higher timeframe trend.
# Targets 15-30 trades per year on 12h timeframe with position size 0.25.

name = "12h_Donchian_20_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for Donchian and volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume spike confirmation
        vol_spike = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, 1-day uptrend, volume spike
            if close[i] > donchian_high[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, 1-day downtrend, volume spike
            elif close[i] < donchian_low[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR 1-day trend turns down
            if close[i] < donchian_low[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR 1-day trend turns up
            if close[i] > donchian_high[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals