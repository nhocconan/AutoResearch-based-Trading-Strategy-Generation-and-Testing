#!/usr/bin/env python3
"""
Hypothesis: 12h Elder Ray Index with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d data).
- Long when Bull Power > 0 and rising (2-bar rising), Short when Bear Power < 0 and falling (2-bar falling).
- Trend filter: Only trade in direction of 1d EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying bull power strength in uptrend, in bear via selling bear power strength in downtrend.
- Uses EMA smoothing which adapts faster than SMMA while reducing noise vs raw price.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, span):
    """Exponential Moving Average with proper min_periods"""
    if len(values) < span:
        return np.full_like(values, np.nan, dtype=float)
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA13 for Elder Ray calculation
    ema_13_1d = ema(close_1d, 13)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components to 12h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = ema(close_1d, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
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
                    # Bull power positive and rising (current > previous)
                    if (bull_power_1d_aligned[i] > 0 and 
                        i > 0 and not np.isnan(bull_power_1d_aligned[i-1]) and
                        bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1] and
                        volume_spike[i]):
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Bear power negative and falling (current < previous)
                    if (bear_power_1d_aligned[i] < 0 and 
                        i > 0 and not np.isnan(bear_power_1d_aligned[i-1]) and
                        bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1] and
                        volume_spike[i]):
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull power becomes non-positive or stops rising
            if (bull_power_1d_aligned[i] <= 0 or 
                i > 0 and not np.isnan(bull_power_1d_aligned[i-1]) and
                bull_power_1d_aligned[i] <= bull_power_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear power becomes non-negative or stops falling
            if (bear_power_1d_aligned[i] >= 0 or 
                i > 0 and not np.isnan(bear_power_1d_aligned[i-1]) and
                bear_power_1d_aligned[i] >= bear_power_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0