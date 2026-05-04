#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Uses Donchian channels from prior completed 6h for structure, 1w EMA200 for major trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# 1w EMA200 provides strong trend filter, reducing whipsaw while capturing major moves.
# Donchian breakouts work well in both bull and bear markets when combined with volume and trend filters.

name = "6h_Donchian20_1wEMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get Donchian(20) from prior completed 6h bars (using rolling window on 6h data)
    # We need to calculate Donchian on the 6h timeframe itself
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + price above 1w EMA200 + volume spike
            if close[i] > high_rolling_max[i] and close[i] > ema_200_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + price below 1w EMA200 + volume spike
            elif close[i] < low_rolling_min[i] and close[i] < ema_200_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below 1w EMA200
            donchian_mid = (high_rolling_max[i] + low_rolling_min[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above 1w EMA200
            donchian_mid = (high_rolling_max[i] + low_rolling_min[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals