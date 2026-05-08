#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1d ADX trend filter
# The Alligator (Jaw/Teeth/Lips) identifies trend direction via SMAs. When Lips cross above Teeth/Jaw,
# it signals an uptrend; cross below signals downtrend. Volume confirms participation, and 1d ADX > 25
# ensures we only trade in strong trends, avoiding whipsaws in ranges. Works in bull/bear by filtering
# for strong trends only. Targets 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe (use previous day's values to avoid look-ahead)
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike detection on 1d (2-day MA)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=2, min_periods=2).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
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
    adx_strong_12h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_12h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure sufficient data for Alligator
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(vol_spike_12h[i]) or np.isnan(adx_strong_12h[i]) or np.isnan(adx_weak_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw, volume spike, strong trend
            if lips_12h[i] > teeth_12h[i] and teeth_12h[i] > jaw_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw, volume spike, strong trend
            elif lips_12h[i] < teeth_12h[i] and teeth_12h[i] < jaw_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips < Teeth or trend weakens
            if lips_12h[i] < teeth_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips > Teeth or trend weakens
            if lips_12h[i] > teeth_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals