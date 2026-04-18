#!/usr/bin/env python3
"""
6h_Weekly_Pivot_R1S1_Breakout_Volume_Filter
Strategy: Breakout above weekly R1 or below weekly S1 on 6h timeframe with volume confirmation.
Long: Price breaks above weekly R1 with volume > 1.5x 20-period average.
Short: Price breaks below weekly S1 with volume > 1.5x 20-period average.
Exit: Price reverts to weekly pivot point (PP) or opposite signal.
Weekly pivot calculated from prior week's OHLC. Uses weekly timeframe for pivot levels.
Designed for 6h timeframe: ~15-30 trades/year per symbol (60-120 total over 4 years).
Works in bull/bear via breakout logic and volume filter to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: PP, R1, S1, R2, S2, R3, S3"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points for each week
    pp_weekly = np.full_like(weekly_high, np.nan)
    r1_weekly = np.full_like(weekly_high, np.nan)
    s1_weekly = np.full_like(weekly_high, np.nan)
    
    for i in range(len(weekly_high)):
        pp, r1, s1, _, _, _, _ = calculate_weekly_pivot(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        pp_weekly[i] = pp
        r1_weekly[i] = r1
        s1_weekly[i] = s1
    
    # Align weekly pivot data to 6h timeframe
    pp_aligned = align_ltf_to_hlf(prices, df_weekly, pp_weekly)
    r1_aligned = align_ltf_to_hlf(prices, df_weekly, r1_weekly)
    s1_aligned = align_ltf_to_hlf(prices, df_weekly, s1_weekly)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_long = high[i] > r1_aligned[i]  # Break above R1
        breakout_short = low[i] < s1_aligned[i]  # Break below S1
        
        # Reversion to pivot (exit condition)
        revert_to_pp = (
            (position == 1 and low[i] <= pp_aligned[i]) or
            (position == -1 and high[i] >= pp_aligned[i])
        )
        
        if position == 0:
            # Long: breakout above R1 with volume
            if breakout_long and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 with volume
            elif breakout_short and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: revert to PP or short breakout
            if revert_to_pp or breakout_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: revert to PP or long breakout
            if revert_to_pp or breakout_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_R1S1_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0