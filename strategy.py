#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + 1w ADX trend filter
# Williams Alligator uses SMAs (13,8,5) with shifts (8,5,3) to identify trends.
# When jaws (13-bar) > teeth (8-bar) > lips (5-bar) = uptrend; reverse for downtrend.
# 1d volume spike confirms institutional participation in the trend.
# 1w ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# Exits occur when Alligator lines re-cross or trend weakens (ADX < 20).
# Targets 12-37 trades per year (~50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for strong trends only.

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
    
    # Williams Alligator on 12h
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMAs
    def sma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        for i in range(period-1, len(arr)):
            result[i] = np.mean(arr[i-period+1:i+1])
        return result
    
    sma_close = sma(close, jaw_period)
    jaw = np.roll(sma_close, jaw_shift)
    sma_close_teeth = sma(close, teeth_period)
    teeth = np.roll(sma_close_teeth, teeth_shift)
    sma_close_lips = sma(close, lips_period)
    lips = np.roll(sma_close_lips, lips_shift)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = vol_1d > (vol_ma_1d.values * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Get 1w data for ADX trend filter
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
    adx_weak = adx < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + jaw_shift, 30)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: jaws > teeth > lips (uptrend), volume spike, strong trend
            if jaw[i] > teeth[i] and teeth[i] > lips[i] and vol_spike_1d_aligned[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: jaws < teeth < lips (downtrend), volume spike, strong trend
            elif jaw[i] < teeth[i] and teeth[i] < lips[i] and vol_spike_1d_aligned[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines re-cross or trend weakens
            if jaw[i] <= teeth[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines re-cross or trend weakens
            if jaw[i] >= teeth[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals