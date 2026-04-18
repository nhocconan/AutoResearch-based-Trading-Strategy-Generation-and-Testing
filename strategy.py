#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Kumo_Twist
Hypothesis: Use Ichimoku cloud twist (Senkou Span A/B cross) from 1d as primary trend filter, combined with Tenkan/Kijun cross on 6h for entry timing and weekly volatility filter. Ichimoku provides multi-dimensional support/resistance with built-in trend strength via cloud thickness. Twist indicates trend change with less whipsaw than single-line crosses. Works in bull markets by entering longs when price above cloud + TK bullish cross in bullish twist, and in bear markets by entering shorts when price below cloud + TK bearish cross in bearish twist. Weekly volatility filter (ATR ratio) avoids chop. Targets 15-25 trades/year by requiring cloud alignment, TK cross, and volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_10 + min_low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not needed for our logic
    
    # Align Ichimoku components to 6h timeframe (wait for bar close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get weekly data for volatility filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on weekly
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(tr)):
        atr_1w[i] = np.mean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_1w_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate ATR(14) on 6h for volatility ratio
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr_6h[i-13:i+1])
    
    # Volatility ratio: current 6h ATR / weekly ATR (expansion signal)
    vol_ratio = atr_6h / atr_1w_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need Senkou B calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Cloud twist: Senkou A > Senkou B = bullish twist, A < B = bearish twist
        bullish_twist = senkou_a_6h[i] > senkou_b_6h[i]
        bearish_twist = senkou_a_6h[i] < senkou_b_6h[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        price_below_cloud = close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        
        # TK cross: Tenkan > Kijun = bullish cross, Tenkan < Kijun = bearish cross
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Volatility filter: expansion (ratio > 1.0) indicates trending conditions
        vol_expansion = vol_ratio[i] > 1.0
        
        if position == 0:
            # Long entry: price above cloud, TK bullish cross, bullish twist, volatility expansion
            if (price_above_cloud and tk_bullish and bullish_twist and vol_expansion):
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud, TK bearish cross, bearish twist, volatility expansion
            elif (price_below_cloud and tk_bearish and bearish_twist and vol_expansion):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price drops below cloud OR TK bearish cross
            if (not price_above_cloud or tk_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud OR TK bullish cross
            if (not price_below_cloud or tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_Kumo_Twist"
timeframe = "6h"
leverage = 1.0