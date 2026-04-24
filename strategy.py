#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend and Alligator lines.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price > 1w EMA50.
         Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price < 1w EMA50.
- Exit: When Alligator alignment breaks or opposite signal.
- Works in bull via buying alignment in uptrend, in bear via selling alignment in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
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
    
    # Get 1w data for Alligator and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 1w
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(df_1w['close'].values, 13)
    jaw = smma(jaw_raw, 8)
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(df_1w['close'].values, 8)
    teeth = smma(teeth_raw, 5)
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(df_1w['close'].values, 5)
    lips = smma(lips_raw, 3)
    
    # Align 1w indicators to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator alignment signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or opposite signal
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or opposite signal
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0