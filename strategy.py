# US30 Index - 6h Pivot Confluence Strategy
# Strategy Type: Pivot Point Breakout with Volume Confirmation
# Timeframe: 6h
# Hypothesis: Combines daily pivot points with volume surge and trend alignment to capture
# institutional breakouts. Works in bull/bear markets by trading breakouts in direction
# of higher timeframe trend. Uses weekly trend filter to avoid counter-trend trades.
# Target: 25-40 trades/year (100-160 over 4 years) with selective entries.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 = (2 * PP) - Low
    r1 = (2 * pp) - low_1d
    # Support 1 = (2 * PP) - High
    s1 = (2 * pp) - high_1d
    # Resistance 2 = PP + (High - Low)
    r2 = pp + (high_1d - low_1d)
    # Support 2 = PP - (High - Low)
    s2 = pp - (high_1d - low_1d)
    # Resistance 3 = High + 2*(PP - Low)
    r3 = high_1d + 2 * (pp - low_1d)
    # Support 3 = Low - 2*(High - PP)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume analysis: current vs 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 8-20 UTC (avoid low volume periods)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current values
        close_val = prices['close'].iloc[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        
        # Skip if any critical value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg_val) or 
            np.isnan(ema_trend)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume surge filter: at least 1.5x average volume
        volume_surge = vol_val > (vol_avg_val * 1.5)
        
        if position == 0:
            # Long conditions: price above weekly EMA (uptrend), 
            # breaks above R1 with volume surge
            if (close_val > ema_trend and 
                close_val > r1_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below weekly EMA (downtrend),
            # breaks below S1 with volume surge
            elif (close_val < ema_trend and 
                  close_val < s1_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot point or volume dries up
            if (close_val < pp_aligned[i] or 
                vol_val < (vol_avg_val * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot point or volume dries up
            if (close_val > pp_aligned[i] or 
                vol_val < (vol_avg_val * 0.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_Pivot_Confluence_VolumeTrend
# Uses daily pivot points (R1/S1) for entry levels
# Requires volume surge (1.5x 20-period average) for confirmation
# Weekly EMA(20) filter ensures trades align with higher timeframe trend
# Session filter: 8-20 UTC to avoid low-volume Asian session
# Exits when price returns to pivot point or volume drops below 80% of average
# Target: 25-40 trades/year with selective, high-probability breakouts
name = "6h_Pivot_Confluence_VolumeTrend"
timeframe = "6h"
leverage = 1.0