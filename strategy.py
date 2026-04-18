# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation.
Buy when price breaks above upper Donchian(20) with volume > 1.5x 20-period average and ATR(14) < ATR(50) (low volatility environment).
Sell when price breaks below lower Donchian(20) with same volume and volatility filters.
Uses volatility filter to avoid whipsaws in high volatility regimes.
Designed for ~20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full(len(high), np.nan)
    atr[period-1] = np.nanmean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) and ATR(50) on 1d
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_50_1d = calculate_atr(high_1d, low_1d, close_1d, 50)
    
    # Align to 4h timeframe
    atr_14_1d_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_4h = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need ATR(50) calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr_14_1d_4h[i]) or np.isnan(atr_50_1d_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) < ATR(50) (low volatility environment)
        vol_filter = atr_14_1d_4h[i] < atr_50_1d_4h[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and volatility filters
            if close[i] > upper[i] and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and volatility filters
            elif close[i] < lower[i] and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRFilter_Volume"
timeframe = "4h"
leverage = 1.0