#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day Elder Ray (bull/bear power) for trend filtering.
# Uses Alligator's jaw/teeth/lips to identify trend direction and strength, while Elder Ray confirms
# bullish/bearish power from daily timeframe. Works in trending markets (both bull and bear) by
# filtering counter-trend noise. Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13511_6h_alligator_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW = 13    # Blue line
ALLIGATOR_TEETH = 8   # Red line
ALLIGATOR_LIPS = 5    # Green line
ELDER_RAY_PERIOD = 13 # EMA period for Elder Ray
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper Wilder's smoothing equivalent"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines"""
    jaw = calculate_ema(close, jaw_period)
    teeth = calculate_ema(close, teeth_period)
    lips = calculate_ema(close, lips_period)
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = calculate_ema(close, period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Elder Ray for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    bull_power, bear_power = calculate_elder_ray(high_1d, low_1d, close_1d, ELDER_RAY_PERIOD)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator lines
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS)
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS, ELDER_RAY_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator trend conditions
        # Uptrend: Lips > Teeth > Jaw (green > red > blue)
        # Downtrend: Jaw > Teeth > Lips (blue > red > green)
        uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        downtrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray confirmation: Bull Power > 0 and Bear Power < 0 for strength
        strong_bull = bull_power_aligned[i] > 0
        strong_bear = bear_power_aligned[i] < 0
        
        # Entry signals
        if position == 0:
            if uptrend and strong_bull:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif downtrend and strong_bear:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay in long while uptrend and bull power positive
            if uptrend and strong_bull:
                signals[i] = SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stay in short while downtrend and bear power negative
            if downtrend and strong_bear:
                signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
    
    return signals