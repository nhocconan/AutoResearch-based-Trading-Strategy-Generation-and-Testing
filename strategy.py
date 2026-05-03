#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channels identify structural breakouts; trading in direction of 12h EMA50 trend
# with volume confirmation provides high-probability continuation trades. Designed for
# low trade frequency (19-50/year) on 4h timeframe to minimize fee drag. Works in both
# bull and bear markets by only taking breakouts aligned with higher timeframe trend.

name = "4h_Donchian20_12hEMA50_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend filter and volume spike
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_12h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_12h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 12h indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Calculate Donchian(20) channels on primary timeframe (4h)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with volume spike in uptrend
            if close[i] > high_rolling_max[i] and close[i-1] <= high_rolling_max[i-1] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume spike in downtrend
            elif close[i] < low_rolling_min[i] and close[i-1] >= low_rolling_min[i-1] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below 20-period high or trend reverses
            if close[i] < high_rolling_max[i] or ema_50_aligned[i] < ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above 20-period low or trend reverses
            if close[i] > low_rolling_min[i] or ema_50_aligned[i] > ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals