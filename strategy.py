#!/usr/bin/env python3
"""
12h Volume-Weighted Average Price (VWAP) Reversion + Volume Spike + Daily Trend
Long when price crosses above VWAP with volume > 1.5x average and daily close > daily open.
Short when price crosses below VWAP with volume > 1.5x average and daily close < daily open.
Exit when price crosses back below/above VWAP.
Designed for low turnover: ~10-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP for the current period"""
    typical_price = (high + low + close) / 3
    vwap = np.nancumsum(typical_price * volume) / np.nancumsum(volume)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_open = df_1d['open'].values
    
    # Calculate VWAP
    vwap = calculate_vwap(high, low, close, volume)
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / vol_count
            vol_sum -= volume[i - 19]
            vol_count -= 1
    
    # Daily trend: 1 if bullish (close > open), -1 if bearish (close < open)
    daily_bullish = daily_close > daily_open
    daily_bearish = daily_close < daily_open
    
    # Create arrays for alignment
    daily_bullish_arr = daily_bullish.astype(float)
    daily_bearish_arr = daily_bearish.astype(float)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if VWAP or volume MA not ready
        if np.isnan(vwap[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get aligned daily trend values
        daily_bull = align_htf_to_ltf(prices, df_1d, daily_bullish_arr)[i]
        daily_bear = align_htf_to_ltf(prices, df_1d, daily_bearish_arr)[i]
        
        if np.isnan(daily_bull) or np.isnan(daily_bear):
            continue
        
        if position == 0:
            # Long: Price crosses above VWAP, volume spike, daily bullish
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and volume[i] > vol_ma[i] * 1.5 and daily_bull > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Price crosses below VWAP, volume spike, daily bearish
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and volume[i] > vol_ma[i] * 1.5 and daily_bear > 0.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below VWAP
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above VWAP
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_VWAP_Reversion_Volume_DailyTrend"
timeframe = "12h"
leverage = 1.0