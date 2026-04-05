#!/usr/bin/env python3
"""
Experiment #10115: 6h Williams Alligator + Elder Ray + ADX Trend Filter
Hypothesis: Combining Williams Alligator (trend), Elder Ray (bull/bear power), and ADX (trend strength) on 6h timeframe 
provides robust trend-following signals that work in both bull and bear markets. Alligator identifies trend direction,
Elder Ray measures momentum strength, and ADX filters for strong trends only. This reduces whipsaws in ranging markets.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10115_6h_williams_alligator_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # Fix first value
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where(tr_smooth != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw = calculate_ema(close, ALLIGATOR_PERIOD_JAW)
    teeth = calculate_ema(close, ALLIGATOR_PERIOD_TEETH)
    lips = calculate_ema(close, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray Power (using 13-period EMA)
    ema_13 = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, ELDER_RAY_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if ADX not available
        if np.isnan(adx[i]) or np.isnan(atr[i]):
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
        
        # Williams Alligator: Mouth open = trending, Mouth closed = ranging
        # Jaw (slowest), Teeth (medium), Lips (fastest)
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Jaw > Teeth > Lips
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray: Bull power > 0 and Bear power < 0 for strong trends
        # Strong uptrend: Bull power increasing and positive
        # Strong downtrend: Bear power decreasing and negative
        elder_long = bull_power[i] > 0 and (i == start or bull_power[i] > bull_power[i-1])
        elder_short = bear_power[i] < 0 and (i == start or bear_power[i] < bear_power[i-1])
        
        # ADX filter: only trade when trend is strong
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = alligator_long and elder_long and strong_trend
        short_entry = alligator_short and elder_short and strong_trend
        
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