#!/usr/bin/env python3
"""
Experiment #11479: 6h Williams Alligator + Elder Ray with 1d Trend Filter
Hypothesis: Williams Alligator identifies trending vs ranging markets, while Elder Ray measures bull/bear power.
In trending markets (JAW < TEETH < LIPS for uptrend), we take Elder Ray signals. In ranging markets, we fade extremes.
Uses 1d EMA for higher-timeframe trend filter to avoid counter-trend trades. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11479_6h_alligator_elder_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
ELDER_RAY_THRESHOLD = 0.02  # 2% of price
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(high, low, close, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).ewm(span=jaw_period, adjust=False, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price).ewm(span=teeth_period, adjust=False, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price).ewm(span=lips_period, adjust=False, min_periods=lips_period).mean().values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray (Bull Power and Bear Power)"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_daily = calculate_ema(df_daily['close'].values, DAILY_EMA_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    jaw, teeth, lips = calculate_alligator(high, low, close, 
                                          ALLIGATOR_JAW_PERIOD, 
                                          ALLIGATOR_TEETH_PERIOD, 
                                          ALLIGATOR_LIPS_PERIOD)
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ELDER_RAY_PERIOD, DAILY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_daily_aligned[i]):
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
        
        # Alligator trend detection
        # Uptrend: JAW < TEETH < LIPS (all rising)
        # Downtrend: JAW > TEETH > LIPS (all falling)
        # Ranging: otherwise (lines intertwined)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            # Not enough data for Alligator
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        is_uptrend = jaw_val < teeth_val < lips_val
        is_downtrend = jaw_val > teeth_val > lips_val
        is_ranging = not (is_uptrend or is_downtrend)
        
        # Elder Ray signals
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if np.isnan(bull_val) or np.isnan(bear_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Normalize Elder Ray by ATR for consistent threshold
        bull_norm = bull_val / atr[i] if atr[i] > 0 else 0
        bear_norm = bear_val / atr[i] if atr[i] > 0 else 0
        
        # Entry logic based on market regime
        long_entry = False
        short_entry = False
        
        if is_uptrend:
            # In uptrend: buy on bullish Elder Ray strength
            long_entry = bull_norm > ELDER_RAY_THRESHOLD
        elif is_downtrend:
            # In downtrend: sell on bearish Elder Ray strength
            short_entry = bear_norm < -ELDER_RAY_THRESHOLD
        else:  # ranging
            # In ranging: fade extremes (contrarian)
            long_entry = bull_norm < -ELDER_RAY_THRESHOLD  # oversold
            short_entry = bear_norm > ELDER_RAY_THRESHOLD  # overbought
        
        # Higher timeframe trend filter (1d EMA)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Apply HTF filter: only take trades in direction of daily trend
        if is_uptrend or is_downtrend:
            # In trending markets, align with daily trend
            final_long = long_entry and uptrend_daily
            final_short = short_entry and downtrend_daily
        else:
            # In ranging markets, still use daily filter but less strict
            final_long = long_entry
            final_short = short_entry
        
        # Generate signals
        if position == 0:
            if final_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif final_short:
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