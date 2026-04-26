#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeFilter
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) identifies 
institutional buying/selling pressure. Go long when Bull Power > 0 and rising, with 1d uptrend 
and volume confirmation. Go short when Bear Power > 0 and rising, with 1d downtrend and 
volume confirmation. Uses 6h timeframe for institutional moves, 1d for trend filter.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate daily EMA13 for Elder Ray and trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13
    # Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low
    
    # Smooth Bull/Bear Power with EMA(8) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # Trend filter: 1d EMA34 for stronger trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA + 8 for smoothing)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Rising Bull/Bear Power (current > previous)
        bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_rising = bear_power_smooth[i] > bear_power_smooth[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, with 1d uptrend and volume confirmation
            if bull_power_smooth[i] > 0 and bull_rising and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and rising, with 1d downtrend and volume confirmation
            elif bear_power_smooth[i] > 0 and bear_rising and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR 1d trend changes to downtrend
            if bull_power_smooth[i] <= 0 or not uptrend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns negative OR 1d trend changes to uptrend
            if bear_power_smooth[i] <= 0 or not downtrend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0