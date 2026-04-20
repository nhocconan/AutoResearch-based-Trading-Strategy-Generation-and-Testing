#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsAlligator_Trend_Confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d: Williams Alligator (SMMA) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2.0  # Williams uses median price
    
    # SMMA (Smoothed Moving Average) calculation
    # Jaw (13-period, 8-shift)
    jaw_1d = _smma(median_price_1d, 13, 8)
    # Teeth (8-period, 5-shift)
    teeth_1d = _smma(median_price_1d, 8, 5)
    # Lips (5-period, 3-shift)
    lips_1d = _smma(median_price_1d, 5, 3)
    
    # Align Alligator lines
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # === 6h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        current_close = close[i]
        current_volume = volume[i]
        current_vol_ma = vol_ma20[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips) or 
            np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = current_volume > 1.3 * current_vol_ma
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume
            if lips > teeth and teeth > jaw and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume
            elif lips < teeth and teeth < jaw and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: Alligator lines cross (lips < teeth) OR stop loss
            if lips < teeth or current_close < entry_price - 2.0 * _calculate_atr(prices, i):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator lines cross (lips > teeth) OR stop loss
            if lips > teeth or current_close > entry_price + 2.0 * _calculate_atr(prices, i):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def _smma(data, period, shift):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    
    # First value is simple SMA
    sma = np.mean(data[:period])
    result = np.full_like(data, np.nan, dtype=float)
    result[period-1] = sma
    
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    
    # Apply shift (Williams Alligator shifts the lines forward)
    shifted_result = np.full_like(data, np.nan, dtype=float)
    if shift < len(data):
        shifted_result[shift:] = result[:-shift]
    
    return shifted_result

def _calculate_atr(prices, idx):
    """Calculate ATR(14) up to given index"""
    if idx < 14:
        return 0.0
    
    high = prices['high'].values[:idx+1]
    low = prices['low'].values[:idx+1]
    close = prices['close'].values[:idx+1]
    
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    return atr[-1] if not np.isnan(atr[-1]) else 0.0