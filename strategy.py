#!/usr/bin/env python3
"""
Experiment #10699: 6h Williams Alligator + Elder Ray 12h Trend
Hypothesis: Combines Williams Alligator (6h) for market structure and Elder Ray (12h) for trend strength.
In bull markets: Green Alligator (lips>teeth>jaws) + Bull Power >0 → long
In bear markets: Red Alligator (jaws>teeth>lips) + Bear Power >0 → short
Uses 12h Elder Ray to filter 6s signals, reducing whipsaw in ranging markets.
Target: 75-150 total trades over 4 years (19-38/year) on 6H timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_10699_6h_williams_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAWS = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(median_price, period_jaws, period_teeth, period_lips):
    """Calculate Williams Alligator lines"""
    jaws = pd.Series(median_price).ewm(span=period_jaws*2, adjust=False, min_periods=period_jaws*2).mean().values
    teeth = pd.Series(median_price).ewm(span=period_teeth*2, adjust=False, min_periods=period_teeth*2).mean().values
    lips = pd.Series(median_price).ewm(span=period_lips*2, adjust=False, min_periods=period_lips*2).mean().values
    return jaws, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = ema - low
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
    
    # Load 12h data ONCE before loop for Elder Ray trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Elder Ray on 12h
    bull_power_12h, bear_power_12h = calculate_elder_ray(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        ELDER_RAY_PERIOD
    )
    
    # Align Elder Ray to 6h timeframe
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h (using median price)
    median_price = (high + low) / 2
    jaws, teeth, lips = calculate_alligator(
        median_price, 
        ALLIGATOR_PERIOD_JAWS, 
        ALLIGATOR_PERIOD_TEETH, 
        ALLIGATOR_PERIOD_LIPS
    )
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAWS*2, ELDER_RAY_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Elder Ray not available
        if np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]):
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
        
        # Williams Alligator signals (6h)
        # Green Alligator: lips > teeth > jaws (bullish alignment)
        # Red Alligator: jaws > teeth > lips (bearish alignment)
        green_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        red_alligator = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray signals (12h) - trend strength
        strong_bull_power = bull_power_12h_aligned[i] > 0
        strong_bear_power = bear_power_12h_aligned[i] > 0
        
        # Entry conditions
        long_entry = green_alligator and strong_bull_power
        short_entry = red_alligator and strong_bear_power
        
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