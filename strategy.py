#!/usr/bin/env python3
"""
6h_ADX_DMI_ADXR_Trend_Strength_V1
Hypothesis: ADX > 25 indicates strong trend; +DI > -DI for long, -DI > +DI for short. ADXR smooths ADX to reduce whipsaw. Trend following works in both bull and bear markets by capturing sustained moves. Uses ADXR (Average Directional Movement Index Rating) for smoother trend strength filtering.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def wilders_smoothing(values, period):
    """Wilder's smoothing (RMA equivalent)"""
    result = np.full_like(values, np.nan)
    if len(values) < period:
        return result
    # First value is simple average
    result[period-1] = np.nanmean(values[:period])
    # Subsequent values: Wilder's smoothing
    for i in range(period, len(values)):
        if not np.isnan(result[i-1]) and not np.isnan(values[i]):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def calculate_adx_dmi(high, low, close, period=14):
    """Calculate ADX, +DI, -DI using Wilder's smoothing"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.concatenate([[np.nan], high[:-1]])),
            np.abs(low - np.concatenate([[np.nan], low[:-1]]))
        )
    )
    
    dm_plus = np.where(
        (high - np.concatenate([[np.nan], high[:-1]])) > 
        (np.concatenate([[np.nan], low[:-1]]) - low),
        np.maximum(high - np.concatenate([[np.nan], high[:-1]]), 0),
        0
    )
    
    dm_minus = np.where(
        (np.concatenate([[np.nan], low[:-1]]) - low) > 
        (high - np.concatenate([[np.nan], high[:-1]])),
        np.maximum(np.concatenate([[np.nan], low[:-1]]) - low, 0),
        0
    )
    
    # Handle NaN values
    dm_plus[0] = 0
    dm_minus[0] = 0
    tr[0] = high[0] - low[0]
    
    # Wilder's smoothing
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # Calculate DI+
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    # Calculate DI-
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # Calculate DX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # Calculate ADX (Wilder's smoothing of DX)
    adx = wilders_smoothing(dx, period)
    
    return adx, di_plus, di_minus

def calculate_adxr(adx, period=14):
    """Calculate ADXR (ADX Rating) - smoothed ADX"""
    # ADXR = (ADX today + ADX period days ago) / 2
    adxr = np.full_like(adx, np.nan)
    for i in range(period, len(adx)):
        if not np.isnan(adx[i]) and not np.isnan(adx[i-period]):
            adxr[i] = (adx[i] + adx[i-period]) / 2
    return adxr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX, +DI, -DI on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    adx_1w, di_plus_1w, di_minus_1w = calculate_adx_dmi(high_1w, low_1w, close_1w, period=14)
    adxr_1w = calculate_adxr(adx_1w, period=14)
    
    # Align weekly indicators to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    di_plus_1w_aligned = align_htf_to_ltf(prices, df_1w, di_plus_1w)
    di_minus_1w_aligned = align_htf_to_ltf(prices, df_1w, di_minus_1w)
    adxr_1w_aligned = align_htf_to_ltf(prices, df_1w, adxr_1w)
    
    # 6h price data
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if NaN in critical values
        if np.isnan(adxr_1w_aligned[i]) or np.isnan(di_plus_1w_aligned[i]) or np.isnan(di_minus_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adxr = adxr_1w_aligned[i]
        di_plus = di_plus_1w_aligned[i]
        di_minus = di_minus_1w_aligned[i]
        
        # Trend strength filter: ADXR > 25 indicates strong trend
        strong_trend = adxr > 25
        
        if position == 0:
            # Long: strong trend + +DI > -DI
            if strong_trend and di_plus > di_minus:
                signals[i] = 0.25
                position = 1
            # Short: strong trend + -DI > +DI
            elif strong_trend and di_minus > di_plus:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens or DI crossover
            if not (adxr > 20 and di_plus > di_minus):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens or DI crossover
            if not (adxr > 20 and di_minus > di_plus):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_DMI_ADXR_Trend_Strength_V1"
timeframe = "6h"
leverage = 1.0