#!/usr/bin/env python3
"""
6h_ElderRay_1dTrend_VolumeBreakout
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation (>1.8x 24-bar avg). 
Long when Bull Power > 0 in 1d uptrend with volume spike. Short when Bear Power < 0 in 1d downtrend with volume spike.
Uses EMA13 for signal smoothing and discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year.
Works in bull/bear by following 1d trend filter and fading extreme Elder Ray readings.
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
    
    # 1d data for HTF trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components on 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume spike: current volume > 1.8x 24-period average (4d equivalent on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 24-period data for volume MA and 50 for 1d EMA
    start_idx = max(24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) in 1d uptrend with volume confirmation
            long_condition = (bull_power_1d_aligned[i] > 0) and \
                            (close_1d[i] > ema_50_1d_aligned[i]) and \
                            volume_spike[i]
            
            # Short: Bear Power < 0 (strong selling) in 1d downtrend with volume confirmation
            short_condition = (bear_power_1d_aligned[i] < 0) and \
                             (close_1d[i] < ema_50_1d_aligned[i]) and \
                             volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR trend turns down OR volume dries up
            if (bull_power_1d_aligned[i] <= 0) or \
               (close_1d[i] < ema_50_1d_aligned[i]) or \
               (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR trend turns up OR volume dries up
            if (bear_power_1d_aligned[i] >= 0) or \
               (close_1d[i] > ema_50_1d_aligned[i]) or \
               (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0