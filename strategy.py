# -*- mode: python; coding: utf-8 -*-

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot levels as structural bias, 6h Donchian(20) breakout, and volume confirmation.
# Long when price breaks above 6h Donchian upper band with price above weekly pivot and volume > 1.5x average.
# Short when price breaks below 6h Donchian lower band with price below weekly pivot and volume > 1.5x average.
# Weekly pivot provides trend bias from higher timeframe, reducing false breakouts in ranging markets.
# Works in bull (breakouts with upward bias) and bear (breakouts with downward bias).
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_1wPivot_6hDonchian_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one full week for pivot calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 6h data for Donchian bands
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # 6h Donchian(20) bands
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align weekly pivot levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Align 6h Donchian bands to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian upper band with price above weekly S3 (bullish bias) and volume spike
            if (close[i] > donchian_high_aligned[i] and
                close[i] > s3_1w_aligned[i] and  # Price above weekly support level 3
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: price breaks below 6h Donchian lower band with price below weekly R3 (bearish bias) and volume spike
            elif (close[i] < donchian_low_aligned[i] and
                  close[i] < r3_1w_aligned[i] and  # Price below weekly resistance level 3
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: price breaks below 6h Donchian lower band or max 40 bars held (~10 days)
            if (close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 6h Donchian upper band or max 40 bars held (~10 days)
            if (close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals