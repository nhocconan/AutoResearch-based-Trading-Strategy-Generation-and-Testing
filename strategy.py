#!/usr/bin/env python3
# 6h_ElderRay_Power_1wTrend_Volume
# Hypothesis: 6h chart strategy using Elder Ray (Bull/Bear Power) with 1-week trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13. Combines with 1-week EMA40 trend filter to avoid counter-trend trades.
# Volume spike (2x average) confirms momentum. Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

timeframe = "6h"
name = "6h_ElderRay_Power_1wTrend_Volume"
leverage = 1.0

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
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate EMA40 on 1-week closes for trend filter
    ema_40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike detection: 2x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 40)  # Ensure we have EMA13 and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 with volume spike and 1-week uptrend
            if bull_power[i] > 0 and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_40_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 with volume spike and 1-week downtrend
            elif bear_power[i] < 0 and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_40_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend failure
            if bull_power[i] <= 0 or close[i] < ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or trend failure
            if bear_power[i] >= 0 or close[i] > ema_40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals