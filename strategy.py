#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
- Long when Bull Power > 0 and rising (making higher low) in uptrend (1d EMA34 rising)
- Short when Bear Power < 0 and falling (making lower high) in downtrend (1d EMA34 falling)
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying Bull Power strength in uptrend, in bear via selling Bear Power weakness in downtrend.
- Uses 1d timeframe for HTF alignment to avoid look-ahead bias.
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
    
    # Calculate EMA13 for Elder Ray (1d)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 6h (each 1d bar = 4x 6h bars)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
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
                    # Long when Bull Power > 0 and rising (making higher low)
                    # Check if Bull Power > 0 and current Bull Power > previous Bull Power
                    if bull_power_aligned[i] > 0 and i > 0 and not np.isnan(bull_power_aligned[i-1]):
                        if bull_power_aligned[i] > bull_power_aligned[i-1] and volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                elif ema34_1d_aligned[i] < ema_34_1d_aligned[i-1]:  # Downtrend
                    # Short when Bear Power < 0 and falling (making lower high)
                    # Check if Bear Power < 0 and current Bear Power < previous Bear Power
                    if bear_power_aligned[i] < 0 and i > 0 and not np.isnan(bear_power_aligned[i-1]):
                        if bear_power_aligned[i] < bear_power_aligned[i-1] and volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or stops rising
            if bull_power_aligned[i] <= 0 or (i > 0 and not np.isnan(bull_power_aligned[i-1]) and bull_power_aligned[i] <= bull_power_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or stops falling
            if bear_power_aligned[i] >= 0 or (i > 0 and not np.isnan(bear_power_aligned[i-1]) and bear_power_aligned[i] >= bear_power_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0