#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination
# Uses Alligator (jaw/teeth/lips) for trend direction and Elder Ray (bull/bear power) for momentum confirmation.
# Works in both bull/bear: Alligator filters sideways markets, Elder Ray captures strength in trending moves.
# Target: 60-120 trades over 4 years (15-30/year) with 6h timeframe to balance frequency and reliability.

name = "exp_12911_6h_alligator_elderay_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(median_price, period_jaw, period_teeth, period_lips):
    """Williams Alligator: smoothed medians with future shift"""
    jaw = pd.Series(median_price).rolling(window=period_jaw*2, min_periods=period_jaw*2).mean()
    jaw = jaw.shift(period_jaw//2)  # shift by half period
    teeth = pd.Series(median_price).rolling(window=period_teeth*2, min_periods=period_teeth*2).mean()
    teeth = teeth.shift(period_teeth//2)
    lips = pd.Series(median_price).rolling(window=period_lips*2, min_periods=period_lips*2).mean()
    lips = lips.shift(period_lips//2)
    return jaw.values, teeth.values, lips.values

def calculate_elderray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Elder Ray for trend bias
    bull_power_1d, bear_power_1d = calculate_elderray(high_1d, low_1d, close_1d, ELDER_RAY_PERIOD)
    trend_bias = bull_power_1d - bear_power_1d  # Positive = bullish bias
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2.0
    jaw, teeth, lips = calculate_alligator(median_price, ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS)
    
    # 6h Elder Ray for entry timing
    bull_power, bear_power = calculate_elderray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW*2, ELDER_RAY_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Alligator not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
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
        
        # Alligator condition: aligned jaws (trending market)
        # Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Elder Ray condition: momentum confirmation
        elder_long = bull_power[i] > 0 and bear_power[i] < 0  # Strong bull power, weak bear power
        elder_short = bull_power[i] < 0 and bear_power[i] > 0  # Strong bear power, weak bull power
        
        # Trend filter from 1d: only trade in direction of higher timeframe bias
        trend_filter_long = trend_bias_aligned[i] > 0
        trend_filter_short = trend_bias_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if alligator_long and elder_long and trend_filter_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif alligator_short and elder_short and trend_filter_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals