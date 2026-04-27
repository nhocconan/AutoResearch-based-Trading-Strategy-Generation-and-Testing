#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_Trend_Follow
Hypothesis: Combines Williams Alligator (trend detection) with Elder Ray (bull/bear power) on 12h timeframe, filtered by 1d trend and volume spikes. Alligator identifies trend direction, Elder Ray measures strength, and volume confirms participation. Works in both bull and bear markets by following established trends with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3 periods)
    # Jaw (blue): 13-period SMMA smoothed by 8 periods
    # Teeth (red): 8-period SMMA smoothed by 5 periods  
    # Lips (green): 5-period SMMA smoothed by 3 periods
    def smma(array, period):
        result = np.full_like(array, np.nan, dtype=float)
        if len(array) >= period:
            sma = np.mean(array[:period])
            result[period-1] = sma
            for i in range(period, len(array)):
                result[i] = (result[i-1] * (period-1) + array[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    jaw_8 = smma(jaw, 8)
    teeth_5 = smma(teeth, 5)
    lips_3 = smma(lips, 3)
    
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw_8)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth_5)
    lips_aligned = align_htf_to_ltf(prices, prices, lips_3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + volume + 1d uptrend
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                bull_power[i] > 0 and
                volume_filter[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator inverted (Lips < Teeth < Jaw) + Bear Power < 0 + volume + 1d downtrend
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  bear_power[i] < 0 and
                  volume_filter[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator convergence or Bear Power negative
            if (lips_aligned[i] <= teeth_aligned[i] or 
                bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator convergence or Bull Power positive
            if (lips_aligned[i] >= teeth_aligned[i] or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Trend_Follow"
timeframe = "12h"
leverage = 1.0