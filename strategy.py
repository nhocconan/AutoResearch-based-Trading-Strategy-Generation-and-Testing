#!/usr/bin/env python3
"""
Experiment #11319: 6h Williams Alligator + Elder Ray Momentum with 12h Trend Filter
Hypothesis: Williams Alligator identifies trend phases (sleeping/awakening/eating). 
Elder Ray (Bull/Bear Power) measures trend strength relative to EMA. Combined with 12h trend filter,
this captures strong momentum moves while avoiding whipsaws in ranging markets. Works in bull/bear 
by requiring alignment between short-term momentum and higher timeframe trend.
Target: 75-175 total trades over 4 years (~19-44/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11319_6w_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed SMA (blue line)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed SMA (red line)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed SMA (green line)
ELDER_RAY_EMA_PERIOD = 13   # EMA for Bull/Bear Power calculation
TREND_EMA_PERIOD = 50       # 12h EMA for trend filter
MIN_POWER_THRESHOLD = 0.0   # Minimum Elder Ray power for entry
SIGNAL_SIZE = 0.25          # Position size (25% of capital)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5   # Wider stop for 6h timeframe

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    sma = np.mean(data[:period])
    smma = np.full_like(data, np.nan, dtype=float)
    smma[period-1] = sma
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    ema_12h = calculate_ema(df_12h['close'].values, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three smoothed SMAs
    jaw = calculate_smma(close, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smma(close, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Shift Alligator lines for predictive nature (Williams method)
    jaw = np.roll(jaw, 3)
    teeth = np.roll(teeth, 2)
    lips = np.roll(lips, 1)
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low
    ema_elder = calculate_ema(close, ELDER_RAY_EMA_PERIOD)
    bull_power = high - ema_elder
    bear_power = ema_elder - low
    
    # ATR for stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for Alligator (jaw is slowest) + EMA
    start = max(ALLIGATOR_JAW_PERIOD + 3, ELDER_RAY_EMA_PERIOD, TREND_EMA_PERIOD) + 5
    
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
        
        # Alligator conditions: check if lines are properly aligned (trending)
        # In uptrend: Lips > Teeth > Jaw (green > red > blue)
        # In downtrend: Lips < Teeth < Jaw (green < red < blue)
        alligator_long = (lips[i] > teeth[i] and teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i] and teeth[i] < jaw[i])
        
        # Elder Ray conditions: strong bull/bear power
        strong_bull_power = bull_power[i] > MIN_POWER_THRESHOLD
        strong_bear_power = bear_power[i] > MIN_POWER_THRESHOLD
        
        # 12h trend filter
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry conditions require ALLIGATOR alignment + Elder Ray strength + 12h trend
        long_entry = alligator_long and strong_bull_power and uptrend_12h
        short_entry = alligator_short and strong_bear_power and downtrend_12h
        
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