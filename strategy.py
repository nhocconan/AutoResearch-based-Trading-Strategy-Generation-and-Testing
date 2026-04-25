#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLine_Cross_1dTrendFilter_VolumeSpike
Hypothesis: Trade Elder Ray Bull/Bear Power zero-line crossovers on 6h with 1d EMA50 trend filter and volume spike confirmation. Elder Ray measures bull/bear power relative to EMA13; zero-line crosses indicate momentum shifts. Works in bull/bear via 1d trend filter (only trade in direction of higher timeframe trend). Volume spike confirms breakout strength. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and volume MA (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: Bull Power crosses above zero + 1d uptrend + volume spike
            long_setup = (bull_power[i] > 0 and bull_power[i-1] <= 0) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: Bear Power crosses below zero + 1d downtrend + volume spike
            short_setup = (bear_power[i] < 0 and bear_power[i-1] >= 0) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bear Power crosses below zero (momentum shift) OR 1d trend turns bearish
            if (bear_power[i] < 0 and bear_power[i-1] >= 0) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bull Power crosses above zero (momentum shift) OR 1d trend turns bullish
            if (bull_power[i] > 0 and bull_power[i-1] <= 0) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLine_Cross_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0