#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trendless markets when lines are intertwined.
# Enter when price breaks above/below the Alligator 'mouth' (Lips) with 1d EMA34 trend alignment and volume spike.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts with trend filter.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw (Blue): 13-period SMMA shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA shifted 5 bars ahead
    # Lips (Green): 5-period SMMA shifted 3 bars ahead
    def smma(source, length):
        """Smoothed Moving Average"""
        if length < 1:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CLOSE) / LENGTH
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaw = smma(high, 13)  # Using high for Jaw as per Williams
    teeth = smma(high, 8)  # Using high for Teeth
    lips = smma(high, 5)   # Using high for Lips
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid Alligator values
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA on 6h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: price breaks above Lips in uptrend alignment with volume spike
            if close[i] > lips[i] and lips[i] > teeth[i] and teeth[i] > jaw[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Lips in downtrend alignment with volume spike
            elif close[i] < lips[i] and lips[i] < teeth[i] and teeth[i] < jaw[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Teeth or loses uptrend alignment
            if close[i] < teeth[i] or lips[i] <= teeth[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Teeth or loses downtrend alignment
            if close[i] > teeth[i] or lips[i] >= teeth[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals