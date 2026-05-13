#!/usr/bin/env python3
"""
6h_WeeklyPivot_Rotation_Filter
Hypothesis: Weekly pivot points act as strong support/resistance. Price tends to rotate between S1-R1 in ranging markets and break through R2/S2 in trending markets. 
In bull/bear markets, we fade at R1/S1 when price is outside weekly CPR (central pivot range), and breakout at R2/S2 when inside CPR. Uses 1d trend filter to avoid counter-trend trades.
Target: 15-30 trades/year per symbol. Works in both bull (breakouts) and bear (mean reversion at extremes).
"""

name = "6h_WeeklyPivot_Rotation_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's data
    # We use shift(1) to ensure we only use completed weekly data
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_close = df_1w['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # 1d trend filter (to avoid counter-trend trades)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # CPR (Central Pivot Range) - area between (H+L+C)/3 and (H+L)/2
    weekly_cpr_top = (weekly_high + weekly_low) / 2
    weekly_cpr_bottom = (weekly_high + weekly_low + weekly_close) / 3  # same as pivot
    weekly_cpr_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cpr_top)
    weekly_cpr_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cpr_bottom)
    
    # Determine if price is inside CPR (range) or outside (trend)
    in_cpr = (close >= weekly_cpr_bottom_aligned) & (close <= weekly_cpr_top_aligned)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if weekly data not available yet
        if np.isnan(weekly_pivot_aligned[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        in_cpr_now = in_cpr[i]
        
        # Fade at extremes when outside CPR (range behavior)
        if not in_cpr_now:
            # Price above CPR - look to fade at R1
            if price <= weekly_r1_aligned[i] and uptrend_1d_aligned[i]:
                # Potential sell at R1 in uptrend
                signals[i] = -0.25
            # Price below CPR - look to fade at S1
            elif price >= weekly_s1_aligned[i] and downtrend_1d_aligned[i]:
                # Potential buy at S1 in downtrend
                signals[i] = 0.25
            else:
                signals[i] = 0.0
        else:
            # Inside CPR - look for breakouts at R2/S2 (trend behavior)
            # Breakout above R2
            if price >= weekly_r2_aligned[i] and uptrend_1d_aligned[i]:
                signals[i] = 0.25
            # Breakdown below S2
            elif price <= weekly_s2_aligned[i] and downtrend_1d_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals