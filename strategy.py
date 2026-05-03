#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Donchian breakouts capture momentum in trending markets. 1d EMA34 ensures trades align with higher timeframe trend.
# Volume spike confirms breakout conviction. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.

name = "4h_Donchian20_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period) using data up to current bar
        lookback = min(20, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: price breaks above Donchian high in 1d uptrend with volume spike
            if close[i] > highest_high and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in 1d downtrend with volume spike
            elif close[i] < lowest_low and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or loses 1d uptrend
            if close[i] < lowest_low or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or loses 1d downtrend
            if close[i] > highest_high or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals