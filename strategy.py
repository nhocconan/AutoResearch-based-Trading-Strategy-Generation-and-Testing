#!/usr/bin/env python3
"""
Experiment #12159: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d Trend Filter
Hypothesis: Williams Alligator identifies trend direction via jaw/teeth/lips alignment.
Elder Ray measures bull/bear power behind price moves. Combined with 1d EMA trend filter,
this captures strong momentum moves while avoiding chop. Works in bull (bull power > 0 with
jaws < teeth < lips) and bear (bear power < 0 with jaws > teeth > lips). Target: 75-200 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12159_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Williams Alligator: SMMA (smoothed moving average) of median price"""
    # Median price = (high + low) / 2
    median_price = (close + close) / 2  # Simplified for close-only, but we'll use typical price later
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(series, period):
        sma = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema_close = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema_close
    bear_power = low - ema_close
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Typical price for Alligator (more accurate than just close)
    typical_price = (high + low + close) / 3
    
    jaw, teeth, lips = calculate_alligator(typical_price, ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD)
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ELDER_RAY_PERIOD, TREND_EMA_PERIOD) + 1
    
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
        
        # Alligator conditions (need valid values)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if np.isnan(bull_val) or np.isnan(bear_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: jaws < teeth < lips = uptrend, jaws > teeth > lips = downtrend
        alligator_uptrend = (jaw_val < teeth_val) and (teeth_val < lips_val)
        alligator_downtrend = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Elder Ray: bull power > 0 = bulls in control, bear power < 0 = bears in control
        bulls_in_control = bull_val > 0
        bears_in_control = bear_val < 0
        
        # Trend filter (1d)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = alligator_uptrend and bulls_in_control and uptrend_1d
        short_entry = alligator_downtrend and bears_in_control and downtrend_1d
        
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