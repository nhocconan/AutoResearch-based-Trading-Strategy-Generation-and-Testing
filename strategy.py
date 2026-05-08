#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w volume spike and 1w ADX trend filter
# Camarilla levels identify key support/resistance. Breakouts above R3 or below S3
# indicate strong momentum. 1w volume spike confirms institutional participation.
# 1w ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# This combination works in both bull and bear markets by filtering for strong trends only.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "12h_Camarilla_R3S3_1wVolume_1wADX"
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
    
    # Get 1w data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla multipliers
    R3 = close_1w + 1.1 * (high_1w - low_1w) / 6
    S3 = close_1w - 1.1 * (high_1w - low_1w) / 6
    R4 = close_1w + 1.382 * (high_1w - low_1w) / 2
    S4 = close_1w - 1.382 * (high_1w - low_1w) / 2
    
    # Align Camarilla levels to 12h timeframe (use previous week's levels)
    R3_12h = align_htf_to_ltf(prices, df_1w, R3)
    S3_12h = align_htf_to_ltf(prices, df_1w, S3)
    R4_12h = align_htf_to_ltf(prices, df_1w, R4)
    S4_12h = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume spike detection on 1w (need ~2 weeks for MA)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()  # 24 * 12h = 12d approx
    vol_spike = volume > (vol_ma.values * 2.0)
    
    # ADX trend filter on 1w
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
    adx_strong_12h = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_12h = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(R4_12h[i]) or 
            np.isnan(S4_12h[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(adx_strong_12h[i]) or np.isnan(adx_weak_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, volume spike, strong trend
            if close[i] > R3_12h[i] and vol_spike[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, volume spike, strong trend
            elif close[i] < S3_12h[i] and vol_spike[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 or trend weakens
            if close[i] < S3_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 or trend weakens
            if close[i] > R3_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals