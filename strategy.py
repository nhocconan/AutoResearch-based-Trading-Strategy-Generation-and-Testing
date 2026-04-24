#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA trend.
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) of median price.
- Breakout: Close > Lips (long) or Close < Jaw (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1w EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying breakouts above Alligator in uptrend, in bear via selling breakdowns below Alligator in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's EMA with alpha=1/period"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Median price for Alligator: (high + low) / 2
    median_price = (high + low) / 2
    
    # Get 1w data for Williams Alligator and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:  # Need at least 13 for Alligator Jaw
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1w median price
    median_1w = (df_1w['high'] + df_1w['low']) / 2
    jaw = smma(median_1w.values, 13)   # Jaw: 13-period SMMA
    teeth = smma(median_1w.values, 8)  # Teeth: 8-period SMMA
    lips = smma(median_1w.values, 5)   # Lips: 5-period SMMA
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
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
                # Long breakout: close > Lips and close > 1w EMA50 (uptrend)
                if close[i] > lips_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Jaw and close < 1w EMA50 (downtrend)
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

name = "12h_Williams_Alligator_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0