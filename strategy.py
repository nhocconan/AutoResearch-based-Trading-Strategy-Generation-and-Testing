#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
# Hypothesis: Breakouts above Donchian(20) high/low are traded only in direction of weekly pivot trend
# with volume confirmation. This avoids counter-trend breakouts and works in both bull/bear
# by filtering with weekly pivot sentiment. Target: 12-37 trades/year.

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Prior week's values
    prev_close = np.roll(close_weekly, 1)
    prev_high = np.roll(high_weekly, 1)
    prev_low = np.roll(low_weekly, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly pivot point and support/resistance
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    
    # Align weekly pivot levels to 6h
    pp_6h = align_htf_to_ltf(prices, df_weekly, pp)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 6h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price touches weekly S1 or breaks below Donchian low
            if low[i] <= s1_6h[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches weekly R1 or breaks above Donchian high
            if high[i] >= r1_6h[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian breakout in direction of weekly pivot trend with volume
            if vol_ok:
                # Determine weekly trend: above PP = uptrend, below PP = downtrend
                if close[i] > pp_6h[i]:  # Weekly uptrend bias
                    if high[i] > donchian_high[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Weekly downtrend bias
                    if low[i] < donchian_low[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals