#!/usr/bin/env python3
"""
Experiment #12291: 6h ADX + Williams Alligator with 1d Trend Filter
Hypothesis: Combine ADX trend strength with Williams Alligator (Jaw/Teeth/Lips) for
direction, filtered by 1d EMA trend. ADX > 25 filters ranging markets, Alligator
alignment confirms trend direction, and 1d EMA ensures alignment with higher timeframe.
This should work in both bull and bear markets by capturing strong trends while
avoiding chop. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12291_6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13  # Smoothed median price, 8 periods ahead
ALLIGATOR_TEETH = 8  # Smoothed median price, 5 periods ahead
ALLIGATOR_LIPS = 5   # Smoothed median price, 3 periods ahead
EMA_TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_alligator(median_price, jaw_period, teeth_period, lips_period):
    """Calculate Williams Alligator lines (Smoothed Median Price with offset)"""
    # Jaw: SMMA of median price, 13 periods, then shift 8
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(8)  # shift future values to avoid look-ahead
    
    # Teeth: SMMA of median price, 8 periods, then shift 5
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(5)  # shift future values to avoid look-ahead
    
    # Lips: SMMA of median price, 5 periods, then shift 3
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(3)  # shift future values to avoid look-ahead
    
    return jaw.values, teeth.values, lips.values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, +DM, -DM
    tr_smoothed = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, EMA_TREND_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Alligator lines
    jaw, teeth, lips = calculate_alligator(median_price, ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period - need enough data for Alligator shifts
    start = max(ALLIGATOR_JAW + 8, ALLIGATOR_TEETH + 5, ALLIGATOR_LIPS + 3, 
                ADX_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
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
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # 1d EMA trend filter
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = alligator_long and strong_trend and uptrend_1d
        short_entry = alligator_short and strong_trend and downtrend_1d
        
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