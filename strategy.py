#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour ADX + Williams Alligator with volume confirmation.
# ADX > 25 indicates trending market. Alligator (Jaw/Teeth/Lips) alignment shows trend direction.
# Volume filter ensures institutional participation. Works in trending markets (both bull/bear).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13319_6h_adx_alligator_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13   # Smoothed SMA (13, 8)
ALLIGATOR_TEETH = 8  # Smoothed SMA (8, 5)
ALLIGATOR_LIPS = 5   # Smoothed SMA (5, 3)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def sma(arr, period):
    """Simple moving average"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

def smma(arr, period):
    """Smoothed moving average ( Wilder's smoothing)"""
    return pd.Series(arr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

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
    tr_smooth = smma(tr, period)
    dm_plus_smooth = smma(dm_plus, period)
    dm_minus_smooth = smma(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = smma(dx, period)
    
    return adx, di_plus, di_minus

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = smma(tr, period)
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for HTF context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d, _, _ = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw = smma(sma(high, ALLIGATOR_JAW), ALLIGATOR_JAW)  # Smoothed SMA(13,8)
    teeth = smma(sma(low, ALLIGATOR_TEETH), ALLIGATOR_TEETH)  # Smoothed SMA(8,5)
    lips = smma(sma(close, ALLIGATOR_LIPS), ALLIGATOR_LIPS)  # Smoothed SMA(5,3)
    
    # Volume MA
    volume_ma = sma(volume, VOLUME_MA_PERIOD)
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_ma[i])):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: daily ADX > 25
        strong_trend = adx_1d_aligned[i] > ADX_THRESHOLD
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, reverse = downtrend
        alligator_up = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_down = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Generate signals
        if position == 0:
            if volume_ok and strong_trend and alligator_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and strong_trend and alligator_down:
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