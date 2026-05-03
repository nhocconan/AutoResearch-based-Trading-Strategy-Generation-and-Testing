#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. Using 1d EMA50 as trend filter ensures
# we only trade in the direction of the daily trend. Volume confirmation adds validity.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in both bull and bear markets by following the trend.

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high over past 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 days
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Get 12h data for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above upper channel or below lower channel with volume spike
        breakout_long = close[i] > upper_20_aligned[i] and volume_spike[i]
        breakout_short = close[i] < lower_20_aligned[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above upper channel in 1d uptrend with volume spike
            if breakout_long and ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel in 1d downtrend with volume spike
            elif breakout_short and ema_50_1d_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper channel or loses 1d uptrend
            if close[i] < upper_20_aligned[i] or ema_50_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower channel or loses 1d downtrend
            if close[i] > lower_20_aligned[i] or ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals