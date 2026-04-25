#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrendFilter
Hypothesis: Trade Ichimoku cloud breakouts on 6h timeframe with weekly trend filter (price above/below weekly Kumo) and volume confirmation (>1.8x 24-bar MA). 
Ichimoku provides dynamic support/resistance via Kumo cloud. Weekly trend filter ensures alignment with higher timeframe momentum. 
Volume confirmation adds conviction to breakouts. Discrete sizing 0.25 balances profit and fee drag. 
Target: 60-120 trades over 4 years (~15-30/year) to stay within fee drag limits.
Works in bull/bear: Ichimoku adapts to volatility, weekly trend filter prevents counter-trend trades, volume confirms breakout strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (weekly Kumo)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for breakout signals)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed as they are concurrent)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Weekly Kumo (cloud) for trend filter
    # Weekly Tenkan-sen
    period_tenkan_1w = 9
    high_tenkan_1w = pd.Series(high_1w).rolling(window=period_tenkan_1w, min_periods=period_tenkan_1w).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=period_tenkan_1w, min_periods=period_tenkan_1w).min().values
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    # Weekly Kijun-sen
    period_kijun_1w = 26
    high_kijun_1w = pd.Series(high_1w).rolling(window=period_kijun_1w, min_periods=period_kijun_1w).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=period_kijun_1w, min_periods=period_kijun_1w).min().values
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    # Weekly Senkou Span A
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Weekly Senkou Span B
    period_senkou_b_1w = 52
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=period_senkou_b_1w, min_periods=period_senkou_b_1w).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=period_senkou_b_1w, min_periods=period_senkou_b_1w).min().values
    senkou_b_1w = (high_senkou_b_1w + low_senkou_b_1w) / 2
    
    # Align weekly Kumo to 6h timeframe
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Weekly Kumo top and bottom
    weekly_kumo_top = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    weekly_kumo_bottom = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and volume MA (24)
    start_idx = max(52, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_kumo_top[i]) or np.isnan(weekly_kumo_bottom[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Kumo top and bottom for current 6h bar
        kumo_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above Kumo AND price above weekly Kumo (bullish weekly trend) AND volume confirm
            long_setup = (close[i] > kumo_top) and \
                         (close[i] > weekly_kumo_top[i]) and \
                         volume_confirm[i]
            # Short: price breaks below Kumo AND price below weekly Kumo (bearish weekly trend) AND volume confirm
            short_setup = (close[i] < kumo_bottom) and \
                          (close[i] < weekly_kumo_bottom[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Kumo OR weekly trend turns bearish (price below weekly Kumo bottom)
            if (close[i] < kumo_top and close[i] > kumo_bottom) or \
               (close[i] < weekly_kumo_bottom[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Kumo OR weekly trend turns bullish (price above weekly Kumo top)
            if (close[i] < kumo_top and close[i] > kumo_bottom) or \
               (close[i] > weekly_kumo_top[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrendFilter"
timeframe = "6h"
leverage = 1.0