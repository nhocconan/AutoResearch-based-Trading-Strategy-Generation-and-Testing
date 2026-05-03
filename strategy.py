#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator identifies trending vs ranging markets via smoothed medians.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via upward alignment and in bear markets via downward alignment.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for Jaw as per Alligator definition
    teeth = smma(low, 8)   # Using low for Teeth
    lips = smma(close, 5)  # Using close for Lips
    
    # Align Alligator lines (no additional delay needed as SMMA uses only past data)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume confirmation: 20-period EMA on 12h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMA and Alligator
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up + price above Lips + daily uptrend + volume spike
            if lips_above_teeth and teeth_above_jaw and close[i] > lips_aligned[i] and ema_50_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + price below Lips + daily downtrend + volume spike
            elif lips_below_teeth and teeth_below_jaw and close[i] < lips_aligned[i] and ema_50_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or price crosses below Lips
            if not (lips_above_teeth and teeth_above_jaw) or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or price crosses above Lips
            if not (lips_below_teeth and teeth_below_jaw) or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals