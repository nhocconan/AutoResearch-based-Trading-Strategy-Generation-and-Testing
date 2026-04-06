#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day Elder Ray for trend confirmation.
# Uses Alligator (Jaw/Teeth/Lips) to detect trends and Elder Ray (Bull/Bear Power) 
# to filter entries in the direction of daily trend. Works in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13499_6h_alligator_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(median_price, period_jaw, period_teeth, period_lips):
    """Calculate Williams Alligator lines"""
    jaw = pd.Series(median_price).ewm(span=period_jaw*2, adjust=False, min_periods=period_jaw*2).mean().values
    teeth = pd.Series(median_price).ewm(span=period_teeth*2, adjust=False, min_periods=period_teeth*2).mean().values
    lips = pd.Series(median_price).ewm(span=period_lips*2, adjust=False, min_periods=period_lips*2).mean().values
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
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
    
    # Median price for Alligator
    median_price = (high + low) / 2.0
    
    # Alligator lines
    jaw, teeth, lips = calculate_alligator(
        median_price, 
        ALLIGATOR_PERIOD_JAW, 
        ALLIGATOR_PERIOD_TEETH, 
        ALLIGATOR_PERIOD_LIPS
    )
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(
        ALLIGATOR_PERIOD_JAW*2, 
        ALLIGATOR_PERIOD_TEETH*2, 
        ALLIGATOR_PERIOD_LIPS*2, 
        ELDER_RAY_PERIOD, 
        ATR_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if Alligator not available
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
        
        # Alligator conditions
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Elder Ray conditions (daily trend)
        bullish_daily = bull_power_aligned[i] > 0
        bearish_daily = bear_power_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            # Long: Alligator bullish + daily bullish
            if lips_above_teeth and teeth_above_jaw and bullish_daily:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator bearish + daily bearish
            elif lips_below_teeth and teeth_below_jaw and bearish_daily:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay long while Alligator remains bullish
            if lips_above_teeth and teeth_above_jaw:
                signals[i] = SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stay short while Alligator remains bearish
            if lips_below_teeth and teeth_below_jaw:
                signals[i] = -SIGNAL_SIZE
            else:
                signals[i] = 0.0
                position = 0
    
    return signals