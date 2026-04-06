#!/usr/bin/env python3
"""
Experiment #11939: 6h Williams Alligator + Elder Ray Momentum with 12h Trend Filter
Hypothesis: Williams Alligator identifies trend presence (jaws-teeth-lips alignment), 
Elder Ray measures bull/bear power, and 12h EMA provides trend bias. This combo filters 
false signals in ranging markets while capturing strong trends. Works in bull (bull power 
positive + alignment) and bear (bear power negative + alignment). Target: 50-150 trades 
over 4 years via strict 3-condition entry requiring Alligator alignment + Elder Ray 
extremes + 12h trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11939_6h_alligator_elder_12h_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAWS_PERIOD = 13   # Smoothed with 8-bar offset
ALLIGATOR_TEETH_PERIOD = 8   # Smoothed with 5-bar offset
ALLIGATOR_LIPS_PERIOD = 5    # Smoothed with 3-bar offset
ELDER_RAY_PERIOD = 13        # For EMA13 in power calculation
TREND_EMA_PERIOD = 50        # 12h EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines"""
    # Median price
    median_price = (high + low) / 2.0 if 'high' in locals() else close  # Will be set properly
    
    # Smoothed medians with offsets
    jaws = pd.Series(median_price).ewm(alpha=2/(jaw_period+1), adjust=False).mean()
    jaws = jaws.shift(8)  # 8-bar offset
    
    teeth = pd.Series(median_price).ewm(alpha=2/(teeth_period+1), adjust=False).mean()
    teeth = teeth.shift(5)  # 5-bar offset
    
    lips = pd.Series(median_price).ewm(alpha=2/(lips_period+1), adjust=False).mean()
    lips = lips.shift(3)  # 3-bar offset
    
    return jaws.values, teeth.values, lips.values

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray Bull Power and Bear Power"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power, ema.values

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
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    median_price = (high + low) / 2.0
    jaws_raw = pd.Series(median_price).ewm(alpha=2/(ALLIGATOR_JAWS_PERIOD+1), adjust=False).mean()
    jaws = jaws_raw.shift(8).values
    
    teeth_raw = pd.Series(median_price).ewm(alpha=2/(ALLIGATOR_TEETH_PERIOD+1), adjust=False).mean()
    teeth = teeth_raw.shift(5).values
    
    lips_raw = pd.Series(median_price).ewm(alpha=2/(ALLIGATOR_LIPS_PERIOD+1), adjust=False).mean()
    lips = lips_raw.shift(3).values
    
    # Elder Ray
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (account for Alligator offsets)
    start = max(ALLIGATOR_JAWS_PERIOD + 8, ALLIGATOR_TEETH_PERIOD + 5, 
                ALLIGATOR_LIPS_PERIOD + 3, ELDER_RAY_PERIOD, TREND_EMA_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Alligator alignment check (requires all three values)
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Alligator bullish alignment: Lips > Teeth > Jaws
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaws[i]
        # Alligator bearish alignment: Jaws > Teeth > Lips
        alligator_bearish = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray extremes (using 1.5x std as threshold)
        if i >= 20:  # Need sufficient history for std
            bp_mean = np.nanmean(bull_power[max(0, i-20):i])
            bp_std = np.nanstd(bull_power[max(0, i-20):i])
            br_mean = np.nanmean(bear_power[max(0, i-20):i])
            br_std = np.nanstd(bear_power[max(0, i-20):i])
            
            bull_strong = bull_power[i] > (bp_mean + 1.5 * bp_std) if not np.isnan(bp_std) and bp_std > 0 else False
            bear_strong = bear_power[i] < (br_mean - 1.5 * br_std) if not np.isnan(br_std) and br_std > 0 else False
        else:
            bull_strong = False
            bear_strong = False
        
        # Trend filter (12h)
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = alligator_bullish and bull_strong and uptrend_12h
        short_entry = alligator_bearish and bear_strong and downtrend_12h
        
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