#!/usr/bin/env python3
"""
6h Elder Ray Index with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d data).
- Long when Bull Power > 0 and rising (2-bar momentum) with volume spike in uptrend (1d EMA34 rising).
- Short when Bear Power < 0 and falling (2-bar momentum) with volume spike in downtrend (1d EMA34 falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strength (Bull Power) in uptrend, in bear via selling weakness (Bear Power) in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray) and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Low - EMA13
    
    # Calculate 2-bar momentum for Elder Ray
    bull_power_momentum_1d = bull_power_1d - np.roll(bull_power_1d, 1)
    bear_power_momentum_1d = bear_power_1d - np.roll(bear_power_1d, 1)
    # Set first value to 0 (no prior bar)
    bull_power_momentum_1d[0] = 0.0
    bear_power_momentum_1d[0] = 0.0
    
    # Align Elder Ray and momentum to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    bull_power_momentum_aligned = align_htf_to_ltf(prices, df_1d, bull_power_momentum_1d)
    bear_power_momentum_aligned = align_htf_to_ltf(prices, df_1d, bear_power_momentum_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # EMA34, volume MA, and momentum needs 1 bar
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_momentum_aligned[i]) or np.isnan(bear_power_momentum_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA34 trend
            if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
                ema34_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long when Bull Power > 0 and rising with volume spike
                    if bull_power_aligned[i] > 0 and bull_power_momentum_aligned[i] > 0 and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short when Bear Power < 0 and falling with volume spike
                    if bear_power_aligned[i] < 0 and bear_power_momentum_aligned[i] < 0 and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or momentum turns negative
            if bull_power_aligned[i] <= 0 or bull_power_momentum_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or momentum turns positive
            if bear_power_aligned[i] >= 0 or bear_power_momentum_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0