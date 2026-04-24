#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume spike.
- Primary timeframe: 12h for execution, HTF: 1d for Elder Ray and 1w for Williams Alligator.
- Williams Alligator (13,8,5 SMAs) from 1w: Long when Jaw < Teeth < Lips (bullish alignment),
  Short when Jaw > Teeth > Lips (bearish alignment).
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 1d: 
  Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling.
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying Alligator bullish alignment with Elder Ray confirmation,
  in bear via selling Alligator bearish alignment with Elder Ray confirmation.
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
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams Alligator (13,8,5 SMAs) from 1w
    jaw_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values  # 13-period SMA (slow)
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values   # 8-period SMA (middle)
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values    # 5-period SMA (fast)
    
    # Align Williams Alligator to 12h (each 1w bar = 14x 12h bars approx)
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Elder Ray from 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray to 12h (each 1d bar = 2x 12h bars)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Elder Ray + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or
            np.isnan(lips_1w_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams Alligator alignment + Elder Ray confirmation
            alligator_bullish = jaw_1w_aligned[i] < teeth_1w_aligned[i] < lips_1w_aligned[i]
            alligator_bearish = jaw_1w_aligned[i] > teeth_1w_aligned[i] > lips_1w_aligned[i]
            
            # Elder Ray: Bull Power > 0 and rising, Bear Power < 0 and falling
            if i > 0 and not np.isnan(bull_power_1d_aligned[i-1]) and not np.isnan(bear_power_1d_aligned[i-1]):
                bull_power_rising = bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]
                bear_power_falling = bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]
                
                if alligator_bullish and bull_power_1d_aligned[i] > 0 and bull_power_rising and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif alligator_bearish and bear_power_1d_aligned[i] < 0 and bear_power_falling and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment or Elder Ray turns negative
            if not (jaw_1w_aligned[i] < teeth_1w_aligned[i] < lips_1w_aligned[i]) or bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment or Elder Ray turns positive
            if not (jaw_1w_aligned[i] > teeth_1w_aligned[i] > lips_1w_aligned[i]) or bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dElderRay_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0