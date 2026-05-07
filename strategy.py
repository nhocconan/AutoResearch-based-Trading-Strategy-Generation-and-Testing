#!/usr/bin/env python3
"""
6h_ElderRay_ZoneRecovery_v1
Hypothesis: Use Elder Ray (Bull Power/Bear Power) on 6h to detect momentum exhaustion and recovery zones. 
Long when Bear Power crosses above zero in oversold zone (price near 20-period low) with volume confirmation.
Short when Bull Power crosses below zero in overbought zone (price near 20-period high) with volume confirmation.
Exit when power crosses back through zero. Uses 1d trend filter to align with higher timeframe momentum.
Designed for 6h to capture mean-reversion within trend with low frequency (target 15-30 trades/year).
"""

name = "6h_ElderRay_ZoneRecovery_v1"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13      # Bull Power: High - EMA13
    bear_power = low - ema13       # Bear Power: Low - EMA13
    
    # 20-period high/low for zone detection
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Zone detection: normalized position within 20-period range
    range_20 = high20 - low20
    # Avoid division by zero
    range_20 = np.where(range_20 == 0, 1, range_20)
    position_in_range = (close - low20) / range_20
    
    # Oversold/overbought zones
    oversold = position_in_range < 0.2   # Near 20-period low
    overbought = position_in_range > 0.8  # Near 20-period high
    
    # 1d trend filter: EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 1d trend: price above/below EMA50
    trend_1d_up = close_1d_aligned > ema_50_1d_aligned
    trend_1d_down = close_1d_aligned < ema_50_1d_aligned
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for 1d EMA50 and 20-period indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bear Power crosses above zero in oversold zone with 1d uptrend
            if (bear_power[i] > 0 and bear_power[i-1] <= 0 and  # crossed above zero
                oversold[i] and 
                trend_1d_up[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power crosses below zero in overbought zone with 1d downtrend
            elif (bull_power[i] < 0 and bull_power[i-1] >= 0 and  # crossed below zero
                  overbought[i] and 
                  trend_1d_down[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power crosses back below zero
            if bear_power[i] < 0 and bear_power[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power crosses back above zero
            if bull_power[i] > 0 and bull_power[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals