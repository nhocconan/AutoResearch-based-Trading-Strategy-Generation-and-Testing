#!/usr/bin/env python3
# 1d_weekly_fvg_retest_v1
# Hypothesis: Weekly Fair Value Gap (FVG) retest strategy on daily timeframe.
# Identifies weekly FVGs (imbalances) and enters on retest to the gap with volume confirmation.
# Works in bull/bear markets by fading extreme imbalances that tend to get filled.
# Low frequency: targets 1-2 trades per month per symbol (~12-24/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_fvg_retest_v1"
timeframe = "1d"
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
    
    # Weekly data for FVG identification
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Identify weekly FVGs (Fair Value Gaps)
    # Bullish FVG: weekly_low[i] > weekly_high[i-2] (gap up)
    # Bearish FVG: weekly_high[i] < weekly_low[i-2] (gap down)
    bullish_fvg = weekly_low[2:] > weekly_high[:-2]
    bearish_fvg = weekly_high[2:] < weekly_low[:-2]
    
    # Prepend two False values to align indices
    bullish_fvg = np.concatenate([np.full(2, False), bullish_fvg])
    bearish_fvg = np.concatenate([np.full(2, False), bearish_fvg])
    
    # Gap levels for retest
    bullish_fvg_low = weekly_high[:-2]  # Lower boundary of bullish FVG
    bearish_fvg_high = weekly_low[:-2]  # Upper boundary of bearish FVG
    
    # Prepend two NaN values to align indices
    bullish_fvg_low = np.concatenate([np.full(2, np.nan), bullish_fvg_low])
    bearish_fvg_high = np.concatenate([np.full(2, np.nan), bearish_fvg_high])
    
    # Align weekly FVG data to daily timeframe
    bullish_fvg_aligned = align_htf_to_ltf(prices, df_1w, bullish_fvg.astype(float))
    bearish_fvg_aligned = align_htf_to_ltf(prices, df_1w, bearish_fvg.astype(float))
    bullish_fvg_low_aligned = align_htf_to_ltf(prices, df_1w, bullish_fvg_low)
    bearish_fvg_high_aligned = align_htf_to_ltf(prices, df_1w, bearish_fvg_high)
    
    # Volume confirmation (20-day average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(bullish_fvg_aligned[i]) or np.isnan(bearish_fvg_aligned[i]) or \
           np.isnan(bullish_fvg_low_aligned[i]) or np.isnan(bearish_fvg_high_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes above FVG upper boundary or opposite FVG forms
            if close[i] > bullish_fvg_high_aligned[i] or bearish_fvg_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes below FVG lower boundary or opposite FVG forms
            if close[i] < bearish_fvg_low_aligned[i] or bullish_fvg_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: retest to bullish FVG lower boundary
                if bullish_fvg_aligned[i] and low[i] <= bullish_fvg_low_aligned[i] * 1.005:
                    position = 1
                    signals[i] = 0.25
                # Short entry: retest to bearish FVG upper boundary
                elif bearish_fvg_aligned[i] and high[i] >= bearish_fvg_high_aligned[i] * 0.995:
                    position = -1
                    signals[i] = -0.25
    
    return signals