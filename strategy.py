#!/usr/bin/env python3
"""
6h_Ichimoku_CloudBreakout_1dFilter
Hypothesis: Ichimoku cloud breakout with 1d trend filter (price > EMA50) yields high-probability trades on 6h timeframe. Works in bull/bear markets by only taking breakouts in direction of 1d trend. Uses Senkou Span A/B for cloud, Tenkan/Kijun for entry signal. Targets 15-30 trades/year by requiring cloud breakout + trend alignment + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(senkou, min_periods=senkou).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily timeframe
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # 1d trend filter: price > EMA50 for bullish, price < EMA50 for bearish
        bullish_trend = price > ema50_1d_aligned[i]
        bearish_trend = price < ema50_1d_aligned[i]
        
        # Ichimoku entry signals
        tk_bullish = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud + TK bullish + volume + bullish trend
            if price > cloud_top and tk_bullish and volume_ok and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + TK bearish + volume + bearish trend
            elif price < cloud_bottom and tk_bearish and volume_ok and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below cloud or TK turns bearish
            if price < cloud_bottom or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above cloud or TK turns bullish
            if price > cloud_top or not tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_1dFilter"
timeframe = "6h"
leverage = 1.0