#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly Camarilla pivot direction + volume confirmation
# Uses weekly Camarilla levels (R3/S3, R4/S4) from 1w timeframe to establish directional bias
# (long bias when close > weekly R3, short bias when close < weekly S3).
# Entry triggered on 6h Donchian(20) breakout in direction of weekly bias with volume spike (>1.5x 20-period average).
# Designed for 12-30 trades/year (~50-120 total over 4 years) to minimize fee drag.
# Weekly Camarilla provides structural bias that works in both bull/bear markets by adapting to higher timeframe pivot levels.
# Donchian breakout captures momentum; volume confirmation filters false breakouts.

name = "6h_Donchian20_1wCamarilla_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on prior week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1
    # S3 = Pivot - Range * 1.1
    # R4 = Pivot + Range * 1.2
    # S4 = Pivot - Range * 1.2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    r4_1w = pivot_1w + range_1w * 1.2
    s4_1w = pivot_1w - range_1w * 1.2
    
    # Align weekly Camarilla to 6h timeframe (wait for completed weekly bar)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 6h Donchian(20) - highest high and lowest low over 20 periods
    # Use pandas rolling with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (>1.5x 20-period average)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND close > weekly R3 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > r3_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND close < weekly S3 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < s3_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR weekly bias flips (close < S3)
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR weekly bias flips (close > R3)
            if (close[i] <= donchian_high[i] and close[i] >= donchian_low[i]) or close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals