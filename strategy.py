#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. Weekly pivot (from 1w) provides structural bias:
# - Price above weekly pivot (PP) = bullish bias, look for long breakouts
# - Price below weekly pivot = bearish bias, look for short breakouts
# Volume spike confirms conviction. Designed for 12-30 trades/year on 6h to minimize fee drag.
# Works in both bull and bear markets by aligning breakout direction with higher timeframe structure.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike"
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
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # We use the prior completed week's values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Calculate Donchian channels (20-period)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using data up to current bar
        highest_high = np.max(high[i-19:i+1])  # 20-period high
        lowest_low = np.min(low[i-19:i+1])     # 20-period low
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high
        breakout_short = close[i] < lowest_low
        
        if position == 0:
            # Long: Donchian breakout above weekly pivot with volume spike
            if breakout_long and close[i] > pp_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below weekly pivot with volume spike
            elif breakout_short and close[i] < pp_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian break below midpoint or loses weekly pivot support
            midpoint = (highest_high + lowest_low) / 2.0
            if close[i] < midpoint or close[i] < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian break above midpoint or loses weekly pivot resistance
            midpoint = (highest_high + lowest_low) / 2.0
            if close[i] > midpoint or close[i] > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals