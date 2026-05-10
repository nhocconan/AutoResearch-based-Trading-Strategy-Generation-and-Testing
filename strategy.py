#!/usr/bin/env python3
# 6h_WilliamsAlligator_ElderRay_TrendFilter
# Hypothesis: Combines Williams Alligator (trend identification) with Elder Ray (bull/bear power) on 1d timeframe.
# The Alligator's jaw-teeth-lips alignment filters trend direction, while Elder Ray measures bull/bear strength.
# Only takes long when bull power > 0 and price above Alligator teeth in uptrend alignment.
# Only takes short when bear power > 0 and price below Alligator teeth in downtrend alignment.
# Uses volume confirmation to avoid false signals. Designed for low trade frequency (15-25/year) on 6h timeframe.
# Works in bull markets via trend-following and in bear via strong bear power signals when aligned.

name = "6h_WilliamsAlligator_ElderRay_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs with specific offsets
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (20-period average on 6d = ~5 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13) + 10  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Trend alignment checks
        # Uptrend: Lips > Teeth > Jaw (all aligned upwards)
        uptrend_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Downtrend: Jaw > Teeth > Lips (all aligned downwards)
        downtrend_aligned = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: Uptrend + Bull Power positive + price above teeth + volume
            if uptrend_aligned and bull_power_aligned[i] > 0 and close[i] > teeth_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + Bear Power positive + price below teeth + volume
            elif downtrend_aligned and bear_power_aligned[i] > 0 and close[i] < teeth_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks down or bear power takes over
            if not uptrend_aligned or bear_power_aligned[i] > bull_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks up or bull power takes over
            if not downtrend_aligned or bull_power_aligned[i] > bear_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals