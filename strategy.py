#!/usr/bin/env python3
# 6h_weekly_donchian_pivot_volume_v1
# Hypothesis: 6h strategy combining weekly Donchian breakout (structure) with daily Camarilla pivot levels (mean reversion zones) and volume confirmation.
# Logic: 
#   - Weekly trend direction from Donchian(20) breakout (bullish if price > 20w high, bearish if price < 20w low)
#   - Within that trend, look for mean reversion entries at daily Camarilla S3/R3 levels
#   - Volume confirmation: current volume > 1.5x 20-period average
#   - Exit: price returns to daily pivot point (PP) or weekly Donchian midpoint
#   - Works in bull/bear: trend filter prevents counter-trend trades in strong moves, mean reversion captures retracements
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_pivot_volume_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Donchian trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    high_s_1w = pd.Series(high_1w)
    low_s_1w = pd.Series(low_1w)
    donchian_high = high_s_1w.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s_1w.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance/Support levels
    r3 = pp + (range_1d * 3.0 / 8.0)
    s3 = pp - (range_1d * 3.0 / 8.0)
    
    # Align Camarilla levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP) or weekly Donchian midpoint
            if close[i] <= pp_aligned[i] or close[i] <= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP) or weekly Donchian midpoint
            if close[i] >= pp_aligned[i] or close[i] >= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine weekly trend from Donchian breakout
            weekly_bullish = close[i] > donchian_high_aligned[i]
            weekly_bearish = close[i] < donchian_low_aligned[i]
            
            # Long entry: weekly bullish trend + price at S3 with volume confirmation
            if weekly_bullish and (close[i] <= s3_aligned[i] * 1.005) and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry: weekly bearish trend + price at R3 with volume confirmation
            elif weekly_bearish and (close[i] >= r3_aligned[i] * 0.995) and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals