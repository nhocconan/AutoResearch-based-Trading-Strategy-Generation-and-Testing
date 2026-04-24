#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA50 trend and Williams Alligator (jaw/teeth/lips).
- Williams Alligator: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3) on median price.
- Long when lips > teeth > jaw (bullish alignment) with volume spike and price > EMA50.
- Short when lips < teeth < jaw (bearish alignment) with volume spike and price < EMA50.
- Volume confirmation: current volume > 1.8x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying Alligator alignment in uptrend, in bear via selling alignment in downtrend.
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
    
    # Get 1d data for EMA50 trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 1d: jaw=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3) on median price
    median_1d = (high_1d + low_1d) / 2
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator lines to 12h (each 1d bar = 2x 12h bars)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: lips > teeth > jaw
            bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Bearish Alligator alignment: lips < teeth < jaw
            bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            if bullish and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif bearish and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or opposite signal
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or opposite signal
            if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0