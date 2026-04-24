#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d data).
- Entry: Long when Bull Power > 0 and rising (Bull Power > previous Bull Power) and volume spike.
         Short when Bear Power < 0 and falling (Bear Power < previous Bear Power) and volume spike.
- Trend filter: Only trade in direction of 1d EMA34 (long if close > EMA34, short if close < EMA34).
- Works in bull via buying strong bull power in uptrend, in bear via selling strong bear power in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Elder Ray signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Bull Power > 0 and rising (strong bullish momentum) and uptrend
                if (bull_power_aligned[i] > 0 and 
                    i > start_idx and bull_power_aligned[i] > bull_power_aligned[i-1] and
                    close[i] > ema_34_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling (strong bearish momentum) and downtrend
                elif (bear_power_aligned[i] < 0 and 
                      i > start_idx and bear_power_aligned[i] < bear_power_aligned[i-1] and
                      close[i] < ema_34_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or trend changes
            if bull_power_aligned[i] <= 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or trend changes
            if bear_power_aligned[i] >= 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0