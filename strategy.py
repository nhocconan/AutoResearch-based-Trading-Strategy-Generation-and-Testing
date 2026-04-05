#!/usr/bin/env python3
"""
Experiment #11631: 6h Williams Alligator + Elder Ray with 1d Regime Filter
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend phases. Elder Ray (Bull/Bear Power) measures trend strength. 
Combined with 1d trend filter (price vs EMA50) to avoid counter-trend trades. Works in bull (Alligator alignment + positive Elder Ray) 
and bear (reverse alignment + negative Elder Ray). Target: 80-160 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11631_6w_alligator_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed with 8-period shift
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed with 5-period shift
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed with 3-period shift
ELDER_RAY_PERIOD = 13       # EMA for Bull/Bear Power
REGIME_EMA_PERIOD = 50      # 1d EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator_lines(median_price, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines (SMMA with specified shifts)"""
    def smma(series, period):
        # Smoothed Moving Average: equivalent to EMA with alpha=1/period
        return pd.Series(series).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts as per Williams Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw: 13-period SMMA smoothed 8 bars ahead
    teeth = np.roll(teeth, 5) # Teeth: 8-period SMMA smoothed 5 bars ahead
    lips = np.roll(lips, 3)   # Lips: 5-period SMMA smoothed 3 bars ahead
    
    # Fill rolled values with NaN for invalid periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    return jaw, teeth, lips

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Calculate 1d EMA50 for regime filter
    ema_1d = calculate_ema(df_1d['close'].values, REGIME_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator - uses median price (high+low)/2
    median_price = (high + low) / 2
    jaw, teeth, lips = calculate_alligator_lines(median_price, ALLIGATOR_JAW_PERIOD, 
                                                 ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray - Bull Power = High - EMA, Bear Power = EMA - Low
    ema_eld = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_eld
    bear_power = ema_eld - low
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need enough data for all indicators)
    start = max(ALLIGATOR_JAW_PERIOD + 8, ALLIGATOR_TEETH_PERIOD + 5, 
                ALLIGATOR_LIPS_PERIOD + 3, ELDER_RAY_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
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
        
        # Alligator alignment check (avoid NaN)
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: JAW > TEETH > LIPS = uptrend, JAW < TEETH < LIPS = downtrend
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 = strong trend
        # For long: Bull Power positive AND Bear Power negative (strong uptrend)
        # For short: Bull Power negative AND Bear Power positive (strong downtrend)
        elder_long = bull_power[i] > 0 and bear_power[i] > 0  # Note: both positive means High > EMA > Low
        elder_short = bull_power[i] < 0 and bear_power[i] < 0  # Actually, let's correct this logic
        
        # Correct Elder Ray interpretation:
        # Bull Power = High - EMA (>0 when price above EMA)
        # Bear Power = EMA - Low (>0 when price below EMA)
        # Strong uptrend: Bull Power > 0 (market making higher highs)
        # Strong downtrend: Bear Power > 0 (market making lower lows)
        elder_long = bull_power[i] > 0  # Making higher highs vs EMA
        elder_short = bear_power[i] > 0  # Making lower lows vs EMA
        
        # Regime filter: 1d EMA50
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = alligator_long and elder_long and uptrend_1d
        short_entry = alligator_short and elder_short and downtrend_1d
        
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