#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Alligator jaws.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Breakout: Close > Lips (long) or Close < Jaw (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1d EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying breakouts above Lips in uptrend, in bear via selling breakdowns below Jaw in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA = smoothed moving average)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(df_1d['close'].values, 13)  # Jaw: 13-period SMMA
    jaw_raw = smma(jaw_raw, 8)                 # Smoothed further by 8-period
    
    teeth_raw = smma(df_1d['close'].values, 8)   # Teeth: 8-period SMMA
    teeth_raw = smma(teeth_raw, 5)               # Smoothed further by 5-period
    
    lips_raw = smma(df_1d['close'].values, 5)    # Lips: 5-period SMMA
    lips_raw = smma(lips_raw, 3)                 # Smoothed further by 3-period
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: close > Lips and close > 1d EMA50 (uptrend)
                if close[i] > lips_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Jaw and close < 1d EMA50 (downtrend)
                elif close[i] < jaw_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Alligator mouth (below Teeth) or opposite signal
            if close[i] < teeth_aligned[i]:  # Exit when price falls below Teeth
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Alligator mouth (above Teeth) or opposite signal
            if close[i] > teeth_aligned[i]:  # Exit when price rises above Teeth
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0