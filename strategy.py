#!/usr/bin/env python3
"""
12h_1d_ChaikinMoneyFlow_VolumeBreakout
Hypothesis: Chaikin Money Flow (CMF) > 0.25 indicates strong accumulation, CMF < -0.25 indicates distribution.
Combine with 12-hour price breaking above/below 20-period high/low with volume confirmation.
Works in bull markets via accumulation breakouts and bear markets via distribution breakdowns.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for CMF calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_low = high_daily - low_daily
    # Avoid division by zero
    high_low_safe = np.where(high_low == 0, 1e-10, high_low)
    mfm = ((close_daily - low_daily) - (high_daily - close_daily)) / high_low_safe
    mfv = mfm * volume_daily
    
    # 20-period sums
    mfv_sum = np.zeros_like(mfv)
    vol_sum = np.zeros_like(volume_daily)
    for i in range(len(mfv)):
        if i >= 19:
            mfv_sum[i] = np.sum(mfv[i-19:i+1])
            vol_sum[i] = np.sum(volume_daily[i-19:i+1])
        else:
            mfv_sum[i] = np.sum(mfv[:i+1]) if i > 0 else mfv[i]
            vol_sum[i] = np.sum(volume_daily[:i+1]) if i > 0 else volume_daily[i]
    
    # Avoid division by zero
    vol_sum_safe = np.where(vol_sum == 0, 1e-10, vol_sum)
    cmf = mfv_sum / vol_sum_safe
    
    # Align daily CMF to 12h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_daily, cmf)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout detection
    high_20 = np.full_like(high, np.nan)
    low_20 = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i >= 19:
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
        else:
            high_20[i] = np.max(high[:i+1]) if i > 0 else high[i]
            low_20[i] = np.min(low[:i+1]) if i > 0 else low[i]
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            volume_avg[i] = np.mean(volume[i-19:i+1])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(cmf_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        cmf_val = cmf_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: CMF > 0.25 (accumulation) + price breaks above 20-period high + volume
            if cmf_val > 0.25 and price > high_20[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.25 (distribution) + price breaks below 20-period low + volume
            elif cmf_val < -0.25 and price < low_20[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative OR price returns to 20-period low
            if cmf_val < 0 or price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive OR price returns to 20-period high
            if cmf_val > 0 or price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_ChaikinMoneyFlow_VolumeBreakout"
timeframe = "12h"
leverage = 1.0