#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-day volume confirmation and 1-week trend filter
# The Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# When the Lips (shortest) crosses above the Teeth (middle) and Jaw (longest), it signals an uptrend.
# When the Lips crosses below the Teeth and Jaw, it signals a downtrend.
# We add 1-day volume surge (>2x 24-period average) to confirm institutional participation.
# We use 1-week ADX > 25 to ensure we only trade in strong weekly trends, avoiding whipsaws.
# This combination works in both bull and bear markets by filtering for strong trends only.
# Targets 12-37 trades per year (~50-150 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_1dVolume_1wADX"
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
    
    # Get 1-day data for Williams Alligator and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator: three SMAs of median price
    # Median price = (high + low) / 2
    median_price = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for rolled values
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # Wilder smoothing
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
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    
    # Volume surge: current volume > 2x 24-period average (approx 6 days)
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 24:
            vol_ma_1d[i] = np.mean(volume_1d[i-24:i])
    vol_surge = volume_1d > (vol_ma_1d * 2.0)
    
    # Align indicators to 12h timeframe
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    vol_surge_12h = align_htf_to_ltf(prices, df_1d, vol_surge)
    adx_strong_12h = align_htf_to_ltf(prices, df_1w, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(jaw_12h[i]) or 
            np.isnan(vol_surge_12h[i]) or np.isnan(adx_strong_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment), volume surge, strong weekly trend
            if lips_12h[i] > teeth_12h[i] > jaw_12h[i] and vol_surge_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment), volume surge, strong weekly trend
            elif lips_12h[i] < teeth_12h[i] < jaw_12h[i] and vol_surge_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips crosses below Teeth or weekly trend weakens
            if lips_12h[i] < teeth_12h[i]:  # or not adx_strong_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips crosses above Teeth or weekly trend weakens
            if lips_12h[i] > teeth_12h[i]:  # or not adx_strong_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals