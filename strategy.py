#!/usr/bin/env python3
# 4h_WilliamsAlligator_ElderRay_Trend_Volume
# Hypothesis: Williams Alligator identifies trend direction (jaws/teeth/lips alignment).
# Elder Ray (Bull/Bear Power) confirms momentum behind the trend.
# Combined with volume confirmation to filter false signals.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends)
# by only trading in direction of Alligator + Elder Ray alignment.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_WilliamsAlligator_ElderRay_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMAs
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to RMA/Wilder's MA)
    close_1d = df_1d['close'].values
    sma5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    # SMMA approximation: first value = SMA, then smoothed
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align all 1d indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need sufficient data for Alligator (13+8=21) and Elder Ray (13)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator trend detection
        # Uptrend: Lips > Teeth > Jaw (green > red > blue)
        # Downtrend: Jaw > Teeth > Lips (blue > red > green)
        uptrend = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        downtrend = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # Elder Ray momentum confirmation
        # Bull Power > 0 indicates bullish momentum
        # Bear Power < 0 indicates bearish momentum
        bull_momentum = bull_power_aligned[i] > 0
        bear_momentum = bear_power_aligned[i] < 0
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + bullish momentum + volume
            if uptrend and bull_momentum and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + bearish momentum + volume
            elif downtrend and bear_momentum and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend/momentum breaks
            if not (uptrend and bull_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend/momentum breaks
            if not (downtrend and bear_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals