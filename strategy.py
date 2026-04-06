#!/usr/bin/env python3
"""
Experiment #12135: 6h Williams Alligator + Elder Ray with Weekly Trend Filter
Hypothesis: Williams Alligator identifies trend presence and direction via smoothed SMAs.
Elder Ray measures bull/bear power behind price movements. Combined with weekly trend
filter, this captures strong trending moves while avoiding chop. Works in bull (bull power > 0,
price above jaw) and bear (bear power < 0, price below jaw). Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12135_6h_alligator_elder_ray_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
WEEKLY_TREND_EMA = 21
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(values, period):
    """Smoothed Moving Average (SMMA) - used in Alligator"""
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(values, np.nan, dtype=float)
    if len(values) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(values)):
            smma[i] = (smma[i-1] * (period-1) + values[i]) / period
    return smma

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    ema_1w = calculate_ema(df_1w['close'].values, WEEKLY_TREND_EMA)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three smoothed SMAs
    jaw = calculate_smma(close, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smma(close, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ELDER_RAY_PERIOD, WEEKLY_TREND_EMA) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Alligator conditions: aligned = trending, tangled = ranging
        # Jaw (slowest), Teeth (middle), Lips (fastest)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check if Alligator is aligned (not tangled)
        # Bull alignment: Lips > Teeth > Jaw
        # Bear alignment: Lips < Teeth < Jaw
        bull_aligned = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bear_aligned = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema_1w_aligned[i]
        downtrend_1w = close[i] < ema_1w_aligned[i]
        
        # Entry conditions
        long_entry = bull_aligned and strong_bull_power and uptrend_1w
        short_entry = bear_aligned and strong_bear_power and downtrend_1w
        
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