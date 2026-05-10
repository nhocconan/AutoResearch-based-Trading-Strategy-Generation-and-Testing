#!/usr/bin/env python3
# 6h_Alligator_ElderRay_1dTrend_Combined
# Hypothesis: Combines Williams Alligator (trend phase detection) with Elder Ray (bull/bear power) on 6h timeframe,
# filtered by daily trend. The Alligator identifies trending vs ranging markets via jaw/teeth/lips alignment.
# Elder Ray measures bull power (high - EMA13) and bear power (EMA13 - low) to assess trend strength.
# Long when: Alligator aligned bullish (lips > teeth > jaw), bull power > 0 and rising, and daily uptrend.
# Short when: Alligator aligned bearish (lips < teeth < jaw), bear power > 0 and rising, and daily downtrend.
# This combination avoids whipsaws by requiring both trend confirmation (Alligator) and momentum validation (Elder Ray).
# Works in bull markets via strong uptrend alignment, and in bear markets via strong downtrend alignment.
# Low trade frequency expected due to multi-condition confluence.

name = "6h_Alligator_ElderRay_1dTrend_Combined"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def alligator(close, jaw_period=13, teeth_period=8, lips_period=5, jaw_shift=8, teeth_shift=5, lips_shift=3):
    """Williams Alligator: three smoothed moving averages"""
    def sma(arr, period):
        return pd.Series(arr).rolling(window=period, min_periods=period).mean()
    
    def smma(arr, period, shift):
        # Smoothed Moving Average (SMMA) - similar to EMA but with different smoothing
        sma_vals = sma(arr, period)
        return pd.Series(sma_vals).ewm(alpha=1/period, adjust=False, min_periods=period).mean().shift(shift)
    
    jaw = smma(close, jaw_period, jaw_shift)
    teeth = smma(close, teeth_period, teeth_shift)
    lips = smma(close, lips_period, lips_shift)
    return jaw.values, teeth.values, lips.values

def elder_ray(high, low, close, ema_period=13):
    """Elder Ray Index: Bull Power and Bear Power"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    bull_power = high - ema.values
    bear_power = ema.values - low
    return bull_power, bear_power, ema.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw, teeth, lips = alligator(close)
    
    # Elder Ray on 6h
    bull_power, bear_power, ema13 = elder_ray(high, low, close)
    
    # Slope of bull/bear power (momentum confirmation)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA (34) + Alligator/Elder Ray components
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from daily EMA
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Alligator alignment signals
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray momentum confirmation
        bull_power_rising = bull_power_slope[i] > 0
        bear_power_rising = bear_power_slope[i] > 0
        
        if position == 0:
            # Long: Alligator bullish, bull power positive and rising, daily uptrend
            if alligator_bullish and bull_power[i] > 0 and bull_power_rising and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish, bear power positive and rising, daily downtrend
            elif alligator_bearish and bear_power[i] > 0 and bear_power_rising and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR bull power turns negative
            if not alligator_bullish or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR bear power turns negative
            if not alligator_bearish or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals