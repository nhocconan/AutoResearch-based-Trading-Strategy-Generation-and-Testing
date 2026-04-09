#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: 6h strategy using weekly pivot points from 1w timeframe for structure,
# with 1d volume confirmation and 6h price action for entry timing.
# Weekly pivot levels (PP, R1-R4, S1-S4) act as major support/resistance zones.
# Breakouts above R1 or below S1 with volume confirmation trigger entries.
# In ranging markets (price between R1 and S1), fade at R3/S3 with volume divergence.
# Uses weekly HTF and 1d volume data called ONCE before loop.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
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
    
    # Weekly HTF data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: PP = (H + L + C)/3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Resistance levels
    r1 = (2 * pp) - low_1w
    r2 = pp + (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    r4 = r3 + (high_1w - low_1w)
    # Support levels
    s1 = (2 * pp) - high_1w
    s2 = pp - (high_1w - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    s4 = s3 - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # 1d HTF data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        # We need the current 1d volume value, not just the MA
        # Since we don't have direct access to current 1d volume in 6h loop,
        # we'll use price action and volume spikes in 6h data for confirmation
        
        # Use 6h volume for confirmation (more responsive)
        volume_s = pd.Series(volume)
        volume_ma_6h = volume_s.rolling(window=20, min_periods=20).mean().values
        if np.isnan(volume_ma_6h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * volume_ma_6h[i]
        
        if position == 1:  # Long position
            # Exit: price falls below PP OR weekly S1
            if close[i] < pp_aligned[i] or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above PP OR weekly R1
            if close[i] > pp_aligned[i] or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine market regime: trending vs ranging
            # Trending: price outside R1-S1 range
            # Ranging: price inside R1-S1 range
            if close[i] > r1_aligned[i] or close[i] < s1_aligned[i]:
                # Trending market - breakout continuation
                # Long breakout above R1 with volume
                if close[i] > r1_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown below S1 with volume
                elif close[i] < s1_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            else:
                # Ranging market - mean reversion at extremes
                # Long near S3 with volume confirmation (bullish divergence)
                if close[i] <= s3_aligned[i] and volume_confirmed:
                    # Additional check: price holding above S3 (bullish sign)
                    if i > 100 and close[i] > low[i-5:i+1].min():
                        position = 1
                        signals[i] = 0.25
                # Short near R3 with volume confirmation (bearish divergence)
                elif close[i] >= r3_aligned[i] and volume_confirmed:
                    # Additional check: price holding below R3 (bearish sign)
                    if i > 100 and close[i] < high[i-5:i+1].max():
                        position = -1
                        signals[i] = -0.25
    
    return signals