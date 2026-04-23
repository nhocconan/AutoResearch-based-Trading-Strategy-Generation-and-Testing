#!/usr/bin/env python3
"""
Hypothesis: 6h ADX(14) + Williams Alligator combination with 12h trend filter.
Long when ADX > 25 (trending) AND price > Alligator Jaw (13-period SMMA) AND price > 12h EMA50.
Short when ADX > 25 AND price < Alligator Jaw AND price < 12h EMA50.
Exit when ADX < 20 (range) OR price crosses Alligator Teeth (8-period SMMA).
Uses 12h HTF for trend alignment and Alligator/Jaw/Teeth/Lips from 6h.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.empty_like(source)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator on 6h timeframe
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # ADX(14) calculation
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    plus_dm[1:] = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm[1:] = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_safe
    
    dx = np.where((plus_di + minus_di) == 0, 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 13)  # ema_50_12h, adx, jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_12h_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) AND price > Jaw AND price > 12h EMA50
            if adx_val > 25 and price > jaw_val and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 AND price < Jaw AND price < 12h EMA50
            elif adx_val > 25 and price < jaw_val and price < ema_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: ADX < 20 (range) OR price crosses Teeth (8-period SMMA)
            if adx_val < 20:
                exit_signal = True
            elif position == 1 and price < teeth_val:
                exit_signal = True
            elif position == -1 and price > teeth_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_Alligator_12hEMA50_Trend_Filter"
timeframe = "6h"
leverage = 1.0