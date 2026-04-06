#!/usr/bin/env python3
"""
Experiment #12239: 6h Williams Alligator with 12h Elder Ray and Volume Confirmation
Hypothesis: The Williams Alligator identifies trend presence and direction (jaw/teeth/lips alignment).
The 12h Elder Ray (Bull/Bear Power) confirms trend strength via EMA deviation. Volume filter ensures
institutional participation. This combo avoids whipsaws in sideways markets while capturing trends.
Works in bull (bull power > 0 + aligned gator) and bear (bear power < 0 + aligned gator) markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12239_6h_alligator_12h_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed SMA
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed SMA
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed SMA
ELDER_RAY_EMA_PERIOD = 13   # EMA for Elder Ray
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smma(values, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

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
    
    # Calculate 12h EMA for Elder Ray
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, ELDER_RAY_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator (using SMMA)
    jaw = smma(high, ALLIGATOR_JAW_PERIOD)  # Typically uses median price, but high works for trend
    teeth = smma(high, ALLIGATOR_TEETH_PERIOD)
    lips = smma(high, ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray Components (12h)
    bull_power = close_12h - ema_12h  # Bull Power = High - EMA (using close as proxy)
    bear_power = ema_12h - close_12h  # Bear Power = EMA - Low (using close as proxy)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, ELDER_RAY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
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
        # Skip if any Alligator line is not available
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        gator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        gator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: Bull Power > 0 and increasing, Bear Power < 0 and decreasing
        # Use current vs previous to check momentum
        bull_power_prev = bull_power_aligned[i-1] if i > 0 else 0
        bear_power_prev = bear_power_aligned[i-1] if i > 0 else 0
        
        elder_long = bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_prev
        elder_short = bear_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_prev  # Bear power positive when EMA > Close
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = gator_long and elder_long and volume_ok
        short_entry = gator_short and elder_short and volume_ok
        
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