#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend via EMA50 on 1w timeframe, enter on Camarilla R1/S1 breakout from 1d with volume confirmation on 12h.
# Exit on opposite Camarilla level touch. Designed for low frequency (15-35 trades/year) to avoid fee drag.
# Camarilla levels provide clear support/resistance, weekly EMA ensures trend alignment, volume confirms breakout strength.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the given period.
    Returns R1, R2, R3, R4, S1, S2, S3, S4 arrays.
    Formula based on previous period's high, low, close.
    """
    n = len(close)
    R1 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    R3 = np.full(n, np.nan)
    R4 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    S2 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    S4 = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            # Use first available values for first bar
            high_val = high[i]
            low_val = low[i]
            close_val = close[i]
        else:
            high_val = high[i-1]
            low_val = low[i-1]
            close_val = close[i-1]
        
        # Calculate pivot point
        pivot = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        # Camarilla levels
        R1[i] = close_val + (range_val * 1.1 / 12)
        S1[i] = close_val - (range_val * 1.1 / 12)
        R2[i] = close_val + (range_val * 1.1 / 6)
        S2[i] = close_val - (range_val * 1.1 / 6)
        R3[i] = close_val + (range_val * 1.1 / 4)
        S3[i] = close_val - (range_val * 1.1 / 4)
        R4[i] = close_val + (range_val * 1.1 / 2)
        S4[i] = close_val - (range_val * 1.1 / 2)
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        # Calculate EMA with proper initialization
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])  # Simple average for first value
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from daily data
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align daily Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate volume average (20-period) for confirmation
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            volume_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation AND above weekly EMA50
            if close[i] > R1_1d_aligned[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation AND below weekly EMA50
            elif close[i] < S1_1d_aligned[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price touches or goes below S1 (opposite level)
            if close[i] < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or goes above R1 (opposite level)
            if close[i] > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals