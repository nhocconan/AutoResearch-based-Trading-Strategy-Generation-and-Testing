#!/usr/bin/env python3
# 1d_Weekly_Pivot_Support_Resistance_Bounce
# Hypothesis: Price tends to bounce off weekly pivot support/resistance levels with volume confirmation.
# In both bull and bear markets, price respects weekly pivot levels as key support/resistance.
# Long when: price crosses above weekly pivot support (S1) with volume > 1.5x 20-day average.
# Short when: price crosses below weekly pivot resistance (R1) with volume > 1.5x 20-day average.
# Exit when price returns to weekly pivot point (PP) or shows contrary price action.
# Uses weekly pivot points calculated from prior week's OHLC for predictive levels.

name = "1d_Weekly_Pivot_Support_Resistance_Bounce"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot point calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, S1 = (2*PP) - H, R1 = (2*PP) - L
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3
    s1 = (2 * pp) - weekly_high  # Support level 1
    r1 = (2 * pp) - weekly_low   # Resistance level 1
    
    # Align weekly pivot levels to daily timeframe (already delayed by weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly data + volume MA
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pp_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        # Price cross above S1 (support bounce) or below R1 (resistance rejection)
        cross_above_s1 = (close[i] > s1_aligned[i]) and (close[i-1] <= s1_aligned[i-1])
        cross_below_r1 = (close[i] < r1_aligned[i]) and (close[i-1] >= r1_aligned[i-1])
        
        if position == 0:
            if cross_above_s1 and vol_spike:
                # Long: bounce from weekly support with volume
                signals[i] = 0.25
                position = 1
            elif cross_below_r1 and vol_spike:
                # Short: rejection at weekly resistance with volume
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price returns to pivot point or breaks below support
                if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot point or breaks above resistance
                if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals