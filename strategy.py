#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA21 trend filter and volume confirmation
# Donchian channels identify key breakout levels; breakouts in direction of 12h trend
# with volume spike provide high-probability continuation trades. Designed for low trade frequency
# (19-50/year) on 4h timeframe to minimize fee drag. Works in both bull and bear markets by
# trading breakouts in the direction of the higher timeframe trend.

name = "4h_Donchian20_12hEMA21_VolumeSpike"
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
    
    # Get 12h data for trend filter and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA21 for trend filter
    ema_21 = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_12h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_12h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 12h indicators to 4h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_21_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike in uptrend
            if close[i] > upper_channel[i] and close[i-1] <= upper_channel[i-1] and ema_21_aligned[i] > close[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike in downtrend
            elif close[i] < lower_channel[i] and close[i-1] >= lower_channel[i-1] and ema_21_aligned[i] < close[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below upper channel
            if close[i] < upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above lower channel
            if close[i] > lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals