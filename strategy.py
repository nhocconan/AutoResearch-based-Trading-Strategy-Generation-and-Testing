#!/usr/bin/env python3
"""
4h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels from weekly chart + volume confirmation + daily trend filter provides high-probability entries in both bull and bear markets. The weekly pivot acts as institutional support/resistance, while volume confirms institutional participation and daily trend ensures alignment with higher timeframe momentum. Targets 20-40 trades/year by requiring weekly pivot proximity + volume spike + daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    l3 = close_val + (range_val * 1.1 / 6)
    l4 = close_val + (range_val * 1.1 / 4)
    h3 = close_val - (range_val * 1.1 / 4)
    h4 = close_val - (range_val * 1.1 / 6)
    return l3, l4, h3, h4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize arrays for Camarilla levels
    l3_1w = np.full_like(close_1w, np.nan)
    l4_1w = np.full_like(close_1w, np.nan)
    h3_1w = np.full_like(close_1w, np.nan)
    h4_1w = np.full_like(close_1w, np.nan)
    
    # Calculate Camarilla for each weekly bar
    for i in range(len(close_1w)):
        l3, l4, h3, h4 = calculate_camarilla(high_1w[i], low_1w[i], close_1w[i])
        l3_1w[i] = l3
        l4_1w[i] = l4
        h3_1w[i] = h3
        h4_1w[i] = h4
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Align weekly Camarilla levels to 4h timeframe
    l3_4h = align_htf_to_ltf(prices, df_1w, l3_1w)
    l4_4h = align_htf_to_ltf(prices, df_1w, l4_1w)
    h3_4h = align_htf_to_ltf(prices, df_1w, h3_1w)
    h4_4h = align_htf_to_ltf(prices, df_1w, h4_1w)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(l3_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(h3_4h[i]) or np.isnan(h4_4h[i]) or
            np.isnan(ema50_4h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        vol_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below H3 level OR trend turns down
            if close[i] < h3_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above L3 level OR trend turns up
            if close[i] > l3_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches L4 level + volume + uptrend
            if (abs(close[i] - l4_4h[i]) < 0.001 * close[i] and  # Within 0.1% of L4
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches H4 level + volume + downtrend
            elif (abs(close[i] - h4_4h[i]) < 0.001 * close[i] and  # Within 0.1% of H4
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals