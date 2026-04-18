#!/usr/bin/env python3
"""
12h Donchian(20) breakout with weekly ATR filter and volume confirmation.
Long when price breaks above 12h Donchian high with volume > 1.5x 20-period average and weekly ATR > daily ATR.
Short when price breaks below 12h Donchian low with volume > 1.5x 20-period average and weekly ATR > daily ATR.
Uses weekly volatility regime to avoid false breakouts in low volatility periods.
Designed for 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(high), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (tr[i] + (period - 1) * atr[i-1]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get weekly data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high = np.full(len(high_12h), np.nan)
    donch_low = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        donch_high[i] = np.max(high_12h[i-20:i])
        donch_low[i] = np.min(low_12h[i-20:i])
    
    # Calculate ATR (14-period) on weekly and daily
    atr_14_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_12h, donch_low)
    atr_14_1w_12h = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(atr_14_1w_12h[i]) or np.isnan(atr_14_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: weekly ATR > daily ATR (indicates healthy volatility)
        vol_filter = atr_14_1w_12h[i] > atr_14_1d_12h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and volatility confirmation
            if close[i] > donch_high_12h[i] and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and volatility confirmation
            elif close[i] < donch_low_12h[i] and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < donch_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > donch_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyATRFilter_Volume"
timeframe = "12h"
leverage = 1.0