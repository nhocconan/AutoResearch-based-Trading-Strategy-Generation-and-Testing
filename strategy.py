#!/usr/bin/env python3
"""
Experiment #11199: 6h Williams Alligator + Elder Ray + 12h Trend
Hypothesis: Williams Alligator identifies trend presence and direction, Elder Ray confirms
momentum behind the trend, and 12h trend filter ensures alignment with higher timeframe.
Works in bull (Alligator jaws up, Elder Bull Power positive) and bear (Alligator jaws down,
Elder Bear Power negative) by requiring alignment. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11199_6h_alligator_elder_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ELDER_RAY_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(close, period):
    """Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift)"""
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan, dtype=float)
        for i in range(len(series)):
            if i < period - 1:
                continue
            elif i == period - 1:
                smma_vals[i] = sma[i]
            else:
                smma_vals[i] = (smma_vals[i-1] * (period - 1) + series[i]) / period
        return smma_vals
    
    jaw = smma(close, period)
    teeth = smma(close, period // 2)  # 8-period
    lips = smma(close, period // 3)   # 5-period (approx)
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    ema_12h = calculate_ema(df_12h['close'].values, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD, ELDER_RAY_PERIOD) + 5  # extra for Alligator shifts
    
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
        
        # Alligator conditions: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_long = (not np.isnan(jaw[i]) and not np.isnan(teeth[i]) and not np.isnan(lips[i]) and
                         jaw[i] > teeth[i] and teeth[i] > lips[i])
        alligator_short = (not np.isnan(jaw[i]) and not np.isnan(teeth[i]) and not np.isnan(lips[i]) and
                          jaw[i] < teeth[i] and teeth[i] < lips[i])
        
        # Elder Ray conditions
        elder_long = bull_power[i] > 0  # Bull Power positive
        elder_short = bear_power[i] > 0  # Bear Power positive
        
        # 12h trend filter
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        # Entry conditions: Alligator direction + Elder Ray momentum + 12h trend alignment
        long_entry = alligator_long and elder_long and uptrend_12h
        short_entry = alligator_short and elder_short and downtrend_12h
        
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