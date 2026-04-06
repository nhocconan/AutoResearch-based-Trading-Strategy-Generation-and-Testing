#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with 1d trend filter.
# Uses ADX(14) > 25 to identify trending markets and Williams Alligator (Jaw/Teeth/Lips)
# for entry timing. 1d EMA50 filter ensures trades align with higher timeframe trend.
# Works in bull markets (buy when price > Alligator lips in uptrend) and bear markets
# (sell when price < Alligator lips in downtrend). Target: 50-150 total trades over 4 years.

name = "exp_13411_6h_adx_alligator_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW_PERIOD = 13  # Smoothed SMA(13), shift 8
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed SMA(8), shift 5
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed SMA(5), shift 3
EMA_PERIOD_1D = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(data, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    smma = np.full_like(data, np.nan, dtype=float)
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

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
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= EMA_PERIOD_1D:
        ema_1d[EMA_PERIOD_1D-1] = np.mean(close_1d[:EMA_PERIOD_1D])
        for i in range(EMA_PERIOD_1D, len(close_1d)):
            ema_1d[i] = (ema_1d[i-1] * (EMA_PERIOD_1D-1) + close_1d[i]) / EMA_PERIOD_1D
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Williams Alligator (using SMMA)
    jaw = calculate_smma(close, ALLIGATOR_JAW_PERIOD)
    teeth = calculate_smma(close, ALLIGATOR_TEETH_PERIOD)
    lips = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Shift Alligator lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted values that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ALLIGATOR_JAW_PERIOD+8, ALLIGATOR_TEETH_PERIOD+5, 
                ALLIGATOR_LIPS_PERIOD+3, EMA_PERIOD_1D, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(adx[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx[i] > ADX_THRESHOLD
        
        # Williams Alligator signals: price relative to lips
        price_above_lips = close[i] > lips_shifted[i]
        price_below_lips = close[i] < lips_shifted[i]
        
        # 1d EMA trend filter
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: ADX > 25, price above lips, and 1d uptrend
            if trending and price_above_lips and uptrend_1d:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: ADX > 25, price below lips, and 1d downtrend
            elif trending and price_below_lips and downtrend_1d:
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