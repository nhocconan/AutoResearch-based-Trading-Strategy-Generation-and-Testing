#!/usr/bin/env python3
"""
Experiment #9855: 6h Williams Alligator + Elder Ray + Weekly Trend
Hypothesis: Williams Alligator identifies trending vs ranging markets, Elder Ray measures bull/bear power,
and weekly trend filter ensures trades align with higher timeframe momentum. 
This combination reduces false signals in ranging markets (common in 2022-2024) while capturing 
strong trends. Works in bull markets (strong bull power + price above alligator) and bear markets 
(strong bear power + price below alligator). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9855_6h_williams_alligator_elder_ray_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
WEEKLY_EMA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Williams Alligator: three smoothed SMAs with future shift"""
    # Williams Alligator uses SMAs with specific shifts
    jaw = pd.Series(close).rolling(window=jaw_period, center=False).mean().shift(jaw_period//2).values
    teeth = pd.Series(close).rolling(window=teeth_period, center=False).mean().shift(teeth_period//2).values
    lips = pd.Series(close).rolling(window=lips_period, center=False).mean().shift(lips_period//2).values
    return jaw, teeth, lips

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
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
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_JAW_PERIOD, 
                                          ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period - need enough data for all indicators
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
                ELDER_RAY_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
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
        
        # Alligator alignment: check if all lines are aligned (trending market)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray conditions: strong bull/bear power
        strong_bull_power = bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0
        
        # Weekly trend filter
        above_weekly_ema = close[i] > weekly_ema_aligned[i]
        below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        # Entry conditions
        long_entry = bullish_alignment and strong_bull_power and above_weekly_ema
        short_entry = bearish_alignment and strong_bear_power and below_weekly_ema
        
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