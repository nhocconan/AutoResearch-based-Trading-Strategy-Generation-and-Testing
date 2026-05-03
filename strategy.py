#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends when lines are aligned and separated.
# Jaw (13-period SMMA) acts as trend filter, Teeth (8-period) and Lips (5-period) as entry/exit triggers.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false signals. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via bullish Alligator alignment and in bear markets via bearish alignment.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirmation"
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
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) aka Wilder's MA"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 20-period EMA on 12h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA and Alligator
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Bullish alignment: Lips > Teeth > Jaw (all rising and separated)
        bullish_alignment = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
        # Bearish alignment: Jaw > Teeth > Lips (all falling and separated)
        bearish_alignment = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        
        if position == 0:
            # Long: bullish Alligator alignment with price above Teeth and volume spike
            if bullish_alignment and close[i] > teeth_shifted[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment with price below Teeth and volume spike
            elif bearish_alignment and close[i] < teeth_shifted[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment or price crosses below Jaw
            if bearish_alignment or close[i] < jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment or price crosses above Jaw
            if bullish_alignment or close[i] > jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals