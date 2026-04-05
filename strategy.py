#!/usr/bin/env python3
"""
Experiment #10619: 6h Williams Alligator + Elder Ray + 12h Trend Filter
Hypothesis: Combine Williams Alligator for trend detection, Elder Ray for momentum strength,
and 12h trend filter to avoid counter-trend trades. Works in bull markets (Alligator mouth up),
bear markets (mouth down), and ranges (mouth closed, Elder Ray near zero). Target: 80-120 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10619_6h_williams_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
TREND_FILTER_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_jaw(high, low, period):
    median = (high + low) / 2
    return pd.Series(median).rolling(window=period, min_periods=period).mean().shift(8).values

def calculate_alligator_teeth(high, low, period):
    median = (high + low) / 2
    return pd.Series(median).rolling(window=period, min_periods=period).mean().shift(5).values

def calculate_alligator_lips(high, low, period):
    median = (high + low) / 2
    return pd.Series(median).rolling(window=period, min_periods=period).mean().shift(3).values

def calculate_elder_ray(high, low, close, period):
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power

def calculate_ema(close, period):
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_FILTER_PERIOD)
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw = calculate_alligator_jaw(high, low, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_alligator_teeth(high, low, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_alligator_lips(high, low, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD + 8, ALLIGATOR_TEETH_PERIOD + 5, 
                ALLIGATOR_LIPS_PERIOD + 3, ELDER_RAY_PERIOD, TREND_FILTER_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Alligator conditions: mouth direction
        # Mouth up: lips > teeth > jaw (bullish alignment)
        # Mouth down: lips < teeth < jaw (bearish alignment)
        # Mouth closed: intertwined (no clear trend)
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        mouth_up = (lips_val > teeth_val) and (teeth_val > jaw_val)
        mouth_down = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Elder Ray: bull/bear power strength
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Strong bull: bull power positive and rising
        # Strong bear: bear power negative and falling (more negative)
        strong_bull = bull_val > 0 and (i == start or bull_val > bull_power[i-1])
        strong_bear = bear_val < 0 and (i == start or bear_val < bear_power[i-1])
        
        # Trend filter: price relative to 12h EMA
        above_12h_ema = close[i] > ema_12h_aligned[i]
        below_12h_ema = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = mouth_up and strong_bull and above_12h_ema
        short_entry = mouth_down and strong_bear and below_12h_ema
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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