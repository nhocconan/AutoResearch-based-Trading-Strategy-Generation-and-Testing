#!/usr/bin/env python3
"""
4h 12h EMA Crossover with 1d Volatility Filter and Volume Confirmation
Long when 12h EMA21 crosses above EMA50 with volatility below median and volume above average
Short when 12h EMA21 crosses below EMA50 with volatility below median and volume above average
Designed to capture trending moves during low volatility periods with confirmation filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA21 and EMA50
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 4h
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1d ATR(10) as volatility measure
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate median of ATR for volatility filter
    atr_median = np.nanmedian(atr_10_1d)
    low_volatility = atr_10_1d < atr_median
    
    # Align volatility filter to 4h
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    # Volume confirmation (above 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_above_avg = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(low_volatility_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_fast = ema_21_12h_aligned[i]
        ema_slow = ema_50_12h_aligned[i]
        low_vol = low_volatility_aligned[i]
        vol_confirm = volume_above_avg[i]
        
        if position == 0:
            # Long: EMA21 crosses above EMA50 + low volatility + volume confirmation
            if ema_fast > ema_slow and low_vol and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: EMA21 crosses below EMA50 + low volatility + volume confirmation
            elif ema_fast < ema_slow and low_vol and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA21 crosses below EMA50 or volatility increases
            if ema_fast < ema_slow or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA21 crosses above EMA50 or volatility increases
            if ema_fast > ema_slow or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA_Crossover_VolumeFilter_VolumeConfirm"
timeframe = "4h"
leverage = 1.0