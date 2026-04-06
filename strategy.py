#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX(14) + Williams Alligator (13,8,5) with daily pivot confirmation.
# Uses ADX > 25 for trend strength and Alligator lines for direction, filtered by daily pivot levels.
# In trending markets (ADX > 25): go long when price > Alligator Jaw and above daily pivot,
# short when price < Alligator Jaw and below daily pivot. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13547_6h_adx_alligator_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13   # Smoothed SMA(13)
ALLIGATOR_TEETH = 8  # Smoothed SMA(8)
ALLIGATOR_LIPS = 5   # Smoothed SMA(5)
SMOOTH_METHOD = 3    # Smoothing periods for Alligator lines
PIVOT_LOOKBACK = 1   # Use previous day's pivot
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    return pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smoothed = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

def calculate_pivot(high, low, close):
    """Calculate daily pivot points: P = (H+L+C)/3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = calculate_pivot(high_1d, low_1d, close_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Williams Alligator (SMMA lines)
    jaw = calculate_smma(close, ALLIGATOR_JAW)
    teeth = calculate_smma(close, ALLIGATOR_TEETH)
    lips = calculate_smma(close, ALLIGATOR_LIPS)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(adx[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(jaw[i]):
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
        
        # Trend filter: ADX > 25
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Alligator direction: price vs Jaw
        price_above_jaw = close[i] > jaw[i]
        price_below_jaw = close[i] < jaw[i]
        
        # Pivot filter: price relative to daily pivot
        price_above_pivot = close[i] > pivot_1d_aligned[i]
        price_below_pivot = close[i] < pivot_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            if strong_trend and price_above_jaw and price_above_pivot:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif strong_trend and price_below_jaw and price_below_pivot:
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