#!/usr/bin/env python3

"""
Hypothesis: 6h Chaikin Money Flow (CMF) with 1-week trend filter and volume confirmation.
Long when CMF > 0.1 (accumulation) and weekly trend is up; short when CMF < -0.1 (distribution)
and weekly trend is down. Uses 20-period CMF to avoid whipsaw. Designed for low trade frequency
(12-37 trades/year) by requiring CMF extremes and trend alignment. Works in both bull and bear
markets by following the weekly trend, avoiding counter-trend trades during regime changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20-period)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(high == low, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cmf[i]) or np.isnan(ema34_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.1 (accumulation) + weekly uptrend
            if cmf[i] > 0.1 and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1 (distribution) + weekly downtrend
            elif cmf[i] < -0.1 and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: CMF returns to neutral zone (-0.1 to 0.1) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: CMF < 0.1 or weekly downtrend
                if cmf[i] < 0.1 or ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: CMF > -0.1 or weekly uptrend
                if cmf[i] > -0.1 or ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_CMF_WeeklyTrend"
timeframe = "6h"
leverage = 1.0