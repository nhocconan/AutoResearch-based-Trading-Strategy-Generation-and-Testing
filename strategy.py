#!/usr/bin/env python3
"""
Experiment #11131: 6h Williams Alligator + Elder Ray with 1d Trend Filter
Hypothesis: Williams Alligator identifies trend direction and strength. Elder Ray measures bull/bear power.
Combined with 1d trend filter, this captures strong trends while avoiding whipsaws in chop.
Works in bull (strong uptrends) and bear (strong downtrends) by requiring alignment across timeframes.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11131_6h_alligator_elder_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ELDER_RAY_PERIOD = 13
SMOOTHING_FACTOR = 8
EMA_TREND_PERIOD = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator_lines(price, period, smoothing):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Jaw (Blue) - 13-period SMMA shifted 8 bars
    jaw = pd.Series(price).rolling(window=period, center=False).mean()
    jaw = jaw.rolling(window=smoothing, center=False).mean()
    jaw = jaw.shift(smoothing)
    
    # Teeth (Red) - 8-period SMMA shifted 5 bars
    teeth = pd.Series(price).rolling(window=period-5, center=False).mean()
    teeth = teeth.rolling(window=smoothing, center=False).mean()
    teeth = teeth.shift(smoothing-3)
    
    # Lips (Green) - 5-period SMMA shifted 3 bars
    lips = pd.Series(price).rolling(window=period-8, center=False).mean()
    lips = lips.rolling(window=smoothing, center=False).mean()
    lips = lips.shift(smoothing-5)
    
    return jaw.values, teeth.values, lips.values

def calculate_elder_ray(high, low, close, ema_period):
    """Calculate Elder Ray: Bull Power and Bear Power"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power, ema.values

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
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator_lines(close, ALLIGATOR_PERIOD, SMOOTHING_FACTOR)
    
    # Elder Ray
    bull_power, bear_power, elder_ema = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + SMOOTHING_FACTOR, ELDER_RAY_PERIOD, EMA_TREND_PERIOD) + 5
    
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
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                         lips[i] > teeth[i] and teeth[i] > jaw[i])
        alligator_short = (not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(jaw[i]) and
                          lips[i] < teeth[i] and teeth[i] < jaw[i])
        
        # Elder Ray: Bull Power > 0 and rising, Bear Power < 0 and falling
        elder_long = (not np.isnan(bull_power[i]) and not np.isnan(bear_power[i]) and
                     bull_power[i] > 0 and (i == 0 or bull_power[i] > bull_power[i-1]) and
                     bear_power[i] < 0)
        elder_short = (not np.isnan(bull_power[i]) and not np.isnan(bear_power[i]) and
                      bear_power[i] < 0 and (i == 0 or bear_power[i] < bear_power[i-1]) and
                      bull_power[i] > 0)
        
        # Trend filter (daily)
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Entry conditions - require Alligator alignment AND Elder Ray confirmation
        long_entry = alligator_long and elder_long and uptrend_daily
        short_entry = alligator_short and elder_short and downtrend_daily
        
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