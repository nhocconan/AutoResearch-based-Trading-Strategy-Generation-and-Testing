#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for Alligator lines and EMA trend.
- Williams Alligator: Jaw (EMA13), Teeth (EMA8), Lips (EMA5) of median price.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price > 1d EMA50 (uptrend).
         Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price < 1d EMA50 (downtrend).
- Exit: When Alligator alignment reverses or price crosses 1d EMA50 opposite.
- Works in bull via buying alignment in uptrend, in bear via selling alignment in downtrend.
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Get 1d data for Alligator and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator components on 1d
    # Jaw: EMA13 of median price
    jaw = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values  # using close as proxy for median
    # Teeth: EMA8 of median price
    teeth = pd.Series(df_1d['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Lips: EMA5 of median price
    lips = pd.Series(df_1d['close']).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator alignment signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Lips > Teeth > Jaw (bullish alignment) with uptrend
                if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips < Teeth < Jaw (bearish alignment) with downtrend
                elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks down or price breaks below EMA50
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks up or price breaks above EMA50
            if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0