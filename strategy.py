#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination
# Uses ADX for trend strength and Williams Alligator (3 SMAs) for direction.
# Works in bull/bear because ADX filters weak trends, Alligator confirms direction.
# Target: 80-150 trades over 4 years (20-38/year) with low frequency to minimize fee drag.
# Only trade when ADX > 25 (strong trend) and price is outside Alligator's mouth.

name = "exp_12887_6h_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13  # SMMA(13, 8)
ALLIGATOR_TEETH = 8  # SMMA(8, 5)
ALLIGATOR_LIPS = 5   # SMMA(5, 3)
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - Williams Alligator uses this"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    smma = np.full_like(data, np.nan)
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
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
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    for i in range(period, len(tr)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[plus_di + minus_di == 0] = 0
    
    adx = np.zeros_like(dx)
    adx[2*period-2] = np.mean(dx[:2*period-1])
    for i in range(2*period-1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate ADX on daily timeframe
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    adx_d = calculate_adx(high_d, low_d, close_d, ADX_PERIOD)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d)
    
    # Calculate Williams Alligator on 6h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator lines: SMMA with different periods and shifts
    jaw = smma(high, ALLIGATOR_JAW)  # Jaw (blue) - SMMA(13, 8)
    teeth = smma(low, ALLIGATOR_TEETH)  # Teeth (red) - SMMA(8, 5)
    lips = smma(close, ALLIGATOR_LIPS)  # Lips (green) - SMMA(5, 3)
    
    # Apply shifts as per Williams Alligator definition
    jaw = np.roll(jaw, ALLIGATOR_JAW//2) if ALLIGATOR_JAW//2 > 0 else jaw
    teeth = np.roll(teeth, ALLIGATOR_TEETH//2) if ALLIGATOR_TEETH//2 > 0 else teeth
    lips = np.roll(lips, ALLIGATOR_LIPS//2) if ALLIGATOR_LIPS//2 > 0 else lips
    
    # Calculate ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD*2, ALLIGATOR_JAW*2) + 1
    
    for i in range(start, n):
        # Skip if ADX not available
        if np.isnan(adx_aligned[i]):
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
        
        # ADX trend strength filter
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Alligator signals: price outside mouth
        # Mouth is between teeth and lips
        jaw_val = jaw[i] if not np.isnan(jaw[i]) else 0
        teeth_val = teeth[i] if not np.isnan(teeth[i]) else 0
        lips_val = lips[i] if not np.isnan(lips[i]) else 0
        
        # Avoid division by zero or invalid comparisons
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Price above all lines = bullish
        # Price below all lines = bearish
        # Price between teeth and lips = inside mouth (no trade)
        above_all = close[i] > jaw_val and close[i] > teeth_val and close[i] > lips_val
        below_all = close[i] < jaw_val and close[i] < teeth_val and close[i] < lips_val
        
        # Generate signals
        if position == 0:
            if strong_trend and above_all:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif strong_trend and below_all:
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