#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and trend filter
# Williams Alligator identifies trend direction using smoothed SMAs (Jaw, Teeth, Lips).
# In strong trends, the Alligator lines are aligned and separated (Jaw > Teeth > Lips for uptrend).
# We add 1d volume confirmation to ensure institutional participation and 1d ADX > 25
# to filter for trending markets only, avoiding whipsaws in ranges.
# This combination works in both bull and bear markets by capturing strong trends.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_1dVolume_1dADX"
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
    
    # Get 1d data for Alligator calculation and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for SMAs
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator: Three SMAs with different periods and shifts
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # We calculate on median price (HL/2) as per Williams
    
    median_price = (high_1d + low_1d) / 2
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV * (N-1) + CURRENT) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # 13-period SMMA
    teeth = smma(median_price, 8)  # 8-period SMMA
    lips = smma(median_price, 5)   # 5-period SMMA
    
    # Apply shifts (Williams Alligator shifts the lines forward)
    # Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # After shifting, the first few values become invalid (rolled from end)
    # Set them to NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Trend condition: Alligator lines aligned and separated
    # Uptrend: Lips > Teeth > Jaw (green > red > blue)
    # Downtrend: Jaw > Teeth > Lips (blue > red > green)
    uptrend_aligned = (lips_12h > teeth_12h) & (teeth_12h > jaw_12h)
    downtrend_aligned = (jaw_12h > teeth_12h) & (teeth_12h > lips_12h)
    
    # Volume confirmation: 1d volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_confirm = volume_1d > (vol_ma_20.values * 1.5)
    volume_confirm_12h = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # ADX trend filter on 1d
    # Calculate ADX(14) on daily
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
    adx_strong_12h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(volume_confirm_12h[i]) or np.isnan(adx_strong_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Uptrend aligned, volume confirmation, strong ADX
            if uptrend_aligned[i] and volume_confirm_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend aligned, volume confirmation, strong ADX
            elif downtrend_aligned[i] and volume_confirm_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Trend breaks down or loses confirmation
            if not uptrend_aligned[i] or not volume_confirm_12h[i] or not adx_strong_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Trend breaks up or loses confirmation
            if not downtrend_aligned[i] or not volume_confirm_12h[i] or not adx_strong_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals