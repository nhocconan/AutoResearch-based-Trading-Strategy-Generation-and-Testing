#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation.
In bull markets: price above 1w Kumo (cloud) + TK cross bullish + volume surge = long.
In bear markets: price below 1w Kumo + TK cross bearish + volume surge = short.
Weekly Ichimoku filters out noise and aligns with major trends. Volume confirms breakout strength.
Designed for 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components."""
    n = len(high)
    if n < 52:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.zeros(n)
    period9_low = np.zeros(n)
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
        else:
            period9_high[i] = np.nan
            period9_low[i] = np.nan
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.zeros(n)
    period26_low = np.zeros(n)
    for i in range(n):
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
        else:
            period26_high[i] = np.nan
            period26_low[i] = np.nan
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            if i + 26 < n:
                senkou_a[i + 26] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.zeros(n)
    period52_low = np.zeros(n)
    for i in range(n):
        if i >= 51:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
        else:
            period52_high[i] = np.nan
            period52_low[i] = np.nan
    senkou_b = np.full(n, np.nan)
    for i in range(n):
        if not (np.isnan(period52_high[i]) or np.isnan(period52_low[i])):
            if i + 26 < n:
                senkou_b[i + 26] = (period52_high[i] + period52_low[i]) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Ichimoku and RSI
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku on 1w
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Calculate RSI on 1w
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align to 6h timeframe
    tenkan_1w_6h = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_6h = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_6h = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_6h = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    rsi_14_1w_6h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need sufficient data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_1w_6h[i]) or np.isnan(kijun_1w_6h[i]) or 
            np.isnan(senkou_a_1w_6h[i]) or np.isnan(senkou_b_1w_6h[i]) or 
            np.isnan(rsi_14_1w_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_1w_6h[i], senkou_b_1w_6h[i])
        cloud_bottom = np.minimum(senkou_a_1w_6h[i], senkou_b_1w_6h[i])
        
        # TK cross
        tk_cross_bullish = tenkan_1w_6h[i] > kijun_1w_6h[i]
        tk_cross_bearish = tenkan_1w_6h[i] < kijun_1w_6h[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above cloud, TK cross bullish, RSI not overbought, volume confirmation
            if (close[i] > cloud_top and tk_cross_bullish and 
                rsi_14_1w_6h[i] < 70 and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK cross bearish, RSI not oversold, volume confirmation
            elif (close[i] < cloud_bottom and tk_cross_bearish and 
                  rsi_14_1w_6h[i] > 30 and vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below cloud or TK cross turns bearish
            if close[i] <= cloud_bottom or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK cross turns bullish
            if close[i] >= cloud_top or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1wTK_RSI_Volume"
timeframe = "6h"
leverage = 1.0