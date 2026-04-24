#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA trend and 1d for Alligator jaws/teeth/lips.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Trend filter: Only trade in direction of 1w EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.5x 50-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying when lips cross above teeth in uptrend, in bear via selling when lips cross below teeth in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator components
    jaw = smma(median_1d, 13)  # Jaw: 13-period SMMA
    teeth = smma(median_1d, 8)  # Teeth: 8-period SMMA
    lips = smma(median_1d, 5)   # Lips: 5-period SMMA
    
    # Smooth further as per Alligator definition
    jaw = smma(jaw, 8)
    teeth = smma(teeth, 5)
    lips = smma(lips, 3)
    
    # Align to 12h: use prior 1d's values (already completed bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 50-period volume MA
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA50 trend
            if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
                ema50_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Buy when lips cross above teeth (bullish alignment)
                    if lips_aligned[i] > teeth_aligned[i] and lips_aligned[i-1] <= teeth_aligned[i-1] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Sell when lips cross below teeth (bearish alignment)
                    if lips_aligned[i] < teeth_aligned[i] and lips_aligned[i-1] >= teeth_aligned[i-1] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: lips cross below teeth (loss of bullish alignment) or opposite cross
            if not np.isnan(lips_aligned[i]) and not np.isnan(teeth_aligned[i]):
                if lips_aligned[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: lips cross above teeth (loss of bearish alignment) or opposite cross
            if not np.isnan(lips_aligned[i]) and not np.isnan(teeth_aligned[i]):
                if lips_aligned[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0