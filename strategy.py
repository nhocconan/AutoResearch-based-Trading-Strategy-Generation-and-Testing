#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels identify key breakout levels; breakouts with weekly trend alignment
# and volume spike provide high-probability continuation trades. Designed for low trade frequency
# (7-25/year) on 1d timeframe to minimize fee drag. Works in both bull and bear markets by
# trading breakouts in the direction of the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20_1w = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (2.0 * vol_ema_20_1w)
    
    # Align 1w indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # Calculate Donchian(20) channels from 1d data
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i]) or 
            np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with volume spike in uptrend
            if close[i] > high_roll_20[i] and close[i-1] <= high_roll_20[i-1] and is_uptrend and volume_spike_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with volume spike in downtrend
            elif close[i] < low_roll_20[i] and close[i-1] >= low_roll_20[i-1] and is_downtrend and volume_spike_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below upper Donchian channel
            if close[i] < high_roll_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above lower Donchian channel
            if close[i] > low_roll_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals