#!/usr/bin/env python3
"""
Hypothesis: 12-hour ADX-based trend following with 1-week volatility filter.
Long when ADX > 25 (trending) and +DI > -DI (bullish momentum) and weekly ATR < median (low volatility).
Short when ADX > 25 and -DI > +DI (bearish momentum) and weekly ATR < median.
Exit when ADX < 20 (trend weakens) or DI crossover reverses.
Uses weekly ATR regime filter to avoid whipsaws in high volatility periods.
Designed for low trade frequency by requiring strong trend + low volatility conditions.
Works in both bull and bear markets by following the trend direction with volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data for ATR volatility filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14) for volatility regime
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly ATR median for regime threshold
    atr_median_1w = np.nanmedian(atr_1w[~np.isnan(atr_1w)])
    atr_1w_low_vol = atr_1w < atr_median_1w  # Low volatility regime
    
    # Align weekly low vol regime to 12h
    atr_1w_low_vol_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_low_vol.astype(float))
    
    # Daily data for ADX calculation (using 1d as proxy for higher timeframe trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation on daily data
    # True Range
    tr1_d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d = np.concatenate([[np.nan], tr_d])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for ADX
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or
            np.isnan(atr_1w_low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend), +DI > -DI (bullish), low volatility regime
            if (adx_aligned[i] > 25 and 
                di_plus_aligned[i] > di_minus_aligned[i] and 
                atr_1w_low_vol_aligned[i] > 0.5):  # True (1.0) indicates low vol
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend), -DI > +DI (bearish), low volatility regime
            elif (adx_aligned[i] > 25 and 
                  di_minus_aligned[i] > di_plus_aligned[i] and 
                  atr_1w_low_vol_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (weak trend) OR -DI crosses above +DI
                if (adx_aligned[i] < 20 or 
                    di_minus_aligned[i] > di_plus_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (weak trend) OR +DI crosses above -DI
                if (adx_aligned[i] < 20 or 
                    di_plus_aligned[i] > di_minus_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_ADX_Trend_1wVolatilityFilter"
timeframe = "12h"
leverage = 1.0