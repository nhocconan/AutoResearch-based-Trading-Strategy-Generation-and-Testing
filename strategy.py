#!/usr/bin/env python3
"""
1d_1w_Momentum_Volume_Regime
Strategy: Daily momentum with volume confirmation and weekly regime filter.
Long: Price > daily VWAP + volume > 1.5x 20-day avg + weekly close > weekly open
Short: Price < daily VWAP + volume > 1.5x 20-day avg + weekly close < weekly open
Exit: Price crosses back through daily VWAP
Position size: 0.25
Designed to capture momentum moves aligned with weekly trend in both bull and bear markets.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP for entry/exit
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly data to daily timeframe
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, weekly_open)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Volume confirmation (20-day MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA and weekly data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(weekly_open_aligned[i]) or 
            np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: weekly close > weekly open (bullish week) or < (bearish week)
        weekly_bullish = weekly_close_aligned[i] > weekly_open_aligned[i]
        weekly_bearish = weekly_close_aligned[i] < weekly_open_aligned[i]
        
        # Entry conditions
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: price above VWAP + volume filter + bullish week
            if price_above_vwap and volume_filter and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP + volume filter + bearish week
            elif price_below_vwap and volume_filter and weekly_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below VWAP
            if price_below_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above VWAP
            if price_above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Momentum_Volume_Regime"
timeframe = "1d"
leverage = 1.0