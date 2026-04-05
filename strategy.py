#!/usr/bin/env python3
"""
Experiment #11331: 6h Williams Alligator + Elder Ray with 1d Trend Filter
Hypothesis: Alligator identifies trend state (sleeping/awakening/feeding), Elder Ray measures bull/bear power.
Combined with 1d trend filter, this captures strong trends while avoiding whipsaws in sideways markets.
Works in bull/bear by using 1d trend to filter direction. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11331_6w_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
DAILY_EMA_PERIOD = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Williams Alligator: three SMAs shifted forward"""
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
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
    
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_JAW_PERIOD, 
                                          ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD)
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD + 8, ELDER_RAY_PERIOD, DAILY_EMA_PERIOD) + 1
    
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
        
        # Alligator conditions: jaws < teeth < lips = downtrend, jaws > teeth > lips = uptrend
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Skip if Alligator values not ready
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        alligator_uptrend = jaw_val > teeth_val > lips_val
        alligator_downtrend = jaw_val < teeth_val < lips_val
        
        # Elder Ray conditions
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Skip if Elder Ray values not ready
        if np.isnan(bull_val) or np.isnan(bear_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Elder Ray: bull power > 0 and rising = strong bulls, bear power < 0 and falling = strong bears
        # For entry, we need alignment with Alligator
        bull_strong = bull_val > 0 and (i == start or bull_val > bull_power[i-1])
        bear_strong = bear_val < 0 and (i == start or bear_val < bear_power[i-1])
        
        # Trend filter (daily)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Entry conditions: Alligator direction + Elder Ray strength + daily trend
        long_entry = alligator_uptrend and bull_strong and uptrend_daily
        short_entry = alligator_downtrend and bear_strong and downtrend_daily
        
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