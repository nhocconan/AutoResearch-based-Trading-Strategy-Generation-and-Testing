#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Hypothesis: Ichimoku cloud strategy on 6h timeframe with weekly trend filter works for BTC and ETH in both bull and bear markets. 
The strategy uses: 
1) Weekly trend filter (price above/below weekly Kumo) 
2) Daily Ichimoku TK cross for entry signals 
3) Daily cloud filter (price above/below Kumo) for confirmation 
4) Volume confirmation to avoid low-quality breakouts 
Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past tenkan periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past kijun periods
    highest_kijun = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted kijun periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past senkou periods plotted kijun periods ahead
    highest_senkou = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    lowest_senkou = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((highest_senkou + lowest_senkou) / 2)
    
    # Chikou Span (Lagging Span): Close plotted kijun periods behind
    chikou_span = close  # Will be aligned properly in main function
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Load daily data once for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Weekly Kumo (cloud) boundaries
    weekly_kumo_top = np.maximum(senkou_a_1w, senkou_b_1w)
    weekly_kumo_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align weekly Ichimoku to 6h timeframe
    weekly_kumo_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_kumo_top)
    weekly_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_kumo_bottom)
    weekly_kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    
    # Calculate daily Ichimoku for entry signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Daily Kumo (cloud) boundaries
    daily_kumo_top = np.maximum(senkou_a_1d, senkou_b_1d)
    daily_kumo_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align daily Ichimoku to 6h timeframe
    daily_tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    daily_kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    daily_kumo_top_aligned = align_htf_to_ltf(prices, df_1d, daily_kumo_top)
    daily_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, daily_kumo_bottom)
    daily_chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_kumo_top_aligned[i]) or np.isnan(weekly_kumo_bottom_aligned[i]) or
            np.isnan(daily_tenkan_aligned[i]) or np.isnan(daily_kijun_aligned[i]) or
            np.isnan(daily_kumo_top_aligned[i]) or np.isnan(daily_kumo_bottom_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.3 * vol_ma[i]
        
        # Weekly trend filter: price above/below weekly Kumo
        weekly_uptrend = price > weekly_kumo_top_aligned[i]
        weekly_downtrend = price < weekly_kumo_bottom_aligned[i]
        
        # Daily Ichimoku signals
        tk_cross_bull = daily_tenkan_aligned[i] > daily_kijun_aligned[i]
        tk_cross_bear = daily_tenkan_aligned[i] < daily_kijun_aligned[i]
        
        # Price relative to daily Kumo
        price_above_daily_kumo = price > daily_kumo_top_aligned[i]
        price_below_daily_kumo = price < daily_kumo_bottom_aligned[i]
        
        if position == 0:
            # Long: Weekly uptrend + TK cross bullish + price above daily Kumo + volume
            if weekly_uptrend and tk_cross_bull and price_above_daily_kumo and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + TK cross bearish + price below daily Kumo + volume
            elif weekly_downtrend and tk_cross_bear and price_below_daily_kumo and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Weekly trend reversal or TK cross bearish or price drops below daily Kumo
            if (not weekly_uptrend) or (not tk_cross_bull) or (price <= daily_kumo_bottom_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Weekly trend reversal or TK cross bullish or price rises above daily Kumo
            if (not weekly_downtrend) or (not tk_cross_bear) or (price >= daily_kumo_top_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0