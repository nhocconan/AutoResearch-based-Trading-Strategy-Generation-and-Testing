#!/usr/bin/env python3
"""
6h_ElderRay_2Period_With_1dTrend
Hypothesis: Use Elder Ray index (Bull Power = High - EMA13, Bear Power = Low - EMA13) 
with 2-period smoothing to reduce noise. Trade in direction of smoothed Elder Ray 
when aligned with daily trend (price vs daily EMA50) and volume confirmation. 
This captures institutional buying/selling pressure while filtering for trend regime. 
Designed for 12-25 trades/year to minimize fee drag. Works in bull/bear via daily trend filter.
"""

name = "6h_ElderRay_2Period_With_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Daily EMA13 for Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Get 6h data for Elder Ray calculation (need high/low)
    # We'll calculate EMA13 on 6h closes, then use with 6h high/low
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Smooth Elder Ray with 2-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Get 6h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    # Get daily close for trend alignment
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: positive bull power (buying pressure) in uptrend with volume
            if bull_power_smooth[i] > 0 and trend_up and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: negative bear power (selling pressure) in downtrend with volume
            elif bear_power_smooth[i] < 0 and trend_down and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or trend turns down
            if bull_power_smooth[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive or trend turns up
            if bear_power_smooth[i] >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals