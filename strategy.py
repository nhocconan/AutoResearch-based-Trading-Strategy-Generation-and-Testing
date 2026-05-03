#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
# Trades in direction of higher timeframe trend during breakouts with volume confirmation.
# Designed for low trade frequency (19-50/year) on 4h timeframe to minimize fee drag.
# Works in both bull and bear markets by following 12h trend during 4h breakouts.

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, in 12h uptrend, with volume spike
            if (close[i] > highest_20[i] and 
                ema_50_12h_aligned[i] > close[i] and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, in 12h downtrend, with volume spike
            elif (close[i] < lowest_20[i] and 
                  ema_50_12h_aligned[i] < close[i] and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle or trend changes
            middle_20 = (highest_20[i] + lowest_20[i]) / 2
            if close[i] < middle_20 or ema_50_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle or trend changes
            middle_20 = (highest_20[i] + lowest_20[i]) / 2
            if close[i] > middle_20 or ema_50_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals