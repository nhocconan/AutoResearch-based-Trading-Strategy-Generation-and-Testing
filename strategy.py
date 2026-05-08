#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d volume spike and 1d ADX trend filter
# Williams Alligator uses smoothed moving averages to identify trend direction.
# When Lips cross above Teeth and Jaw, it signals an uptrend; when below, downtrend.
# Volume spike confirms institutional participation. 1d ADX > 25 ensures strong trends.
# This combination filters out whipsaws and works in both bull/bear markets by trading
# only during strong trends. Targets ~20-40 trades/year to minimize fee drag.

name = "4h_WilliamsAlligator_1dVolume_1dADX"
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
    
    # Get 1d data for Williams Alligator and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for SMMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, 8 bars ahead
    # Teeth (red): 8-period SMMA, 5 bars ahead
    # Lips (green): 5-period SMMA, 3 bars ahead
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Align Alligator lines to 4h timeframe
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike detection on 1d (volume > 2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma_20.values * 2.0)
    
    # ADX trend filter on 1d
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_4h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_4h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_4h[i]) or np.isnan(adx_weak_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment), volume spike, strong trend
            if lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment), volume spike, strong trend
            elif lips_4h[i] < teeth_4h[i] and teeth_4h[i] < jaw_4h[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or trend weakens
            if not (lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i]) or adx_weak_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or trend weakens
            if not (lips_4h[i] < teeth_4h[i] and teeth_4h[i] < jaw_4h[i]) or adx_weak_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals