#!/usr/bin/env python3
"""
Experiment #10191: 6h Williams Alligator + Elder Ray + ADX
Hypothesis: Williams Alligator defines trend direction (Green=up, Red=down), Elder Ray measures bull/bear power,
and ADX confirms trend strength (>25). Enter long when bull power > 0, bear power < 0, Alligator green, ADX > 25.
Short when bear power > 0, bull power < 0, Alligator red, ADX > 25. Works in trending markets (both bull/bear)
by capturing strong directional moves with momentum confirmation.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10191_6h_williams_alligator_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_sma(close, period):
    """Calculate SMA"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothing
    tr_sum = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (SMAs of median price)
    median_price = (high + low) / 2
    jaw = calculate_sma(median_price, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_sma(median_price, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_sma(median_price, ALLIGATOR_LIPS_PERIOD)
    
    # Shift as per Williams Alligator rules
    jaw = np.roll(jaw, ALLIGATOR_JAW_PERIOD // 2)
    teeth = np.roll(teeth, ALLIGATOR_TEETH_PERIOD // 2)
    lips = np.roll(lips, ALLIGATOR_LIPS_PERIOD // 2)
    
    # Elder Ray (13-period EMA)
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
    start = max(ALLIGATOR_JAW_PERIOD, ELDER_RAY_PERIOD, ADX_PERIOD) + 10
    
    for i in range(start, n):
        # Skip if any indicator not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]):
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
        
        # Alligator alignment: Green (bullish) when Lips > Teeth > Jaw
        # Red (bearish) when Lips < Teeth < Jaw
        alligator_green = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_red = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: Bull power > 0 and Bear power < 0 for long bias
        # Bear power > 0 and Bull power < 0 for short bias
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
        bearish_momentum = bear_power[i] > 0 and bull_power[i] < 0
        
        # ADX trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Entry conditions
        long_entry = bullish_momentum and alligator_green and strong_trend
        short_entry = bearish_momentum and alligator_red and strong_trend
        
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