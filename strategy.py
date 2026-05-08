#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot (R3/S3) breakout + volume spike + 1w ADX trend filter
# Camarilla pivot levels (R3/S3) act as key intraday support/resistance with high breakout potential.
# Volume spike confirms institutional participation in the breakout.
# 1w ADX > 25 ensures trading only in strong trends, avoiding whipsaws in ranges.
# Exits occur when price returns to the Camarilla pivot point (PP) or trend weakens (ADX < 20).
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for strong trends only.

name = "1d_Camarilla_R3S3_1dVolume_1wADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp = np.full_like(close, np.nan)
    r3 = np.full_like(close, np.nan)
    s3 = np.full_like(close, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC
        pp[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        r3[i] = pp[i] + (high[i-1] - low[i-1]) * 1.1 / 2.0
        s3[i] = pp[i] - (high[i-1] - low[i-1]) * 1.1 / 2.0
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 2.0)
    
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
    
    start_idx = 20  # Ensure sufficient data for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, volume spike, strong trend
            if close[i] > r3[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, volume spike, strong trend
            elif close[i] < s3[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point or trend weakens
            if close[i] < pp[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point or trend weakens
            if close[i] > pp[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals