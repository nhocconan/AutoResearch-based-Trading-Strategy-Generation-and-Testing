#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and 1d volume confirmation
# Donchian channels identify structural breakouts; 12h EMA50 ensures alignment with intermediate trend to avoid whipsaws.
# 1d volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts with trend filter.

name = "6h_Donchian20_12hEMA50_1dVolumeSpike"
timeframe = "6h"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d volume for volume spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start from lookback to have full Donchian calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 * aligned 1d volume EMA20
        volume_spike = volume[i] > (1.5 * vol_ema_20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper band in 12h uptrend with volume spike
            if close[i] > highest_high[i] and ema_50_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band in 12h downtrend with volume spike
            elif close[i] < lowest_low[i] and ema_50_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower band or loses 12h uptrend
            if close[i] < lowest_low[i] or ema_50_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper band or loses 12h downtrend
            if close[i] > highest_high[i] or ema_50_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals