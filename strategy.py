#!/usr/bin/env python3
"""
6h 1D Ichimoku Cloud with 1W Trend Filter
Long: Price above 1D Kumo + TK Cross Bullish + 1W EMA100 Up
Short: Price below 1D Kumo + TK Cross Bearish + 1W EMA100 Down
Exit: Price crosses opposite Kumo edge (Tenkan/Kijun average)
Uses Ichimoku for trend/momentum and weekly EMA for long-term bias
Target: 15-25 trades/year per symbol
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
    
    # Get 1D data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: Tenkan=9, Kijun=26, Senkou B=52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): current close shifted 26 periods back (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Get 1W data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_100_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        # Kumo top and bottom (Senkou Span A and B)
        kumo_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        # Kumo midpoint for exit signal
        kumo_mid = (kumo_top + kumo_bottom) / 2
        
        if position == 0:
            # Long: Price above Kumo + TK Cross Bullish + 1W Uptrend
            if (price > kumo_top and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                ema_100_1w_aligned[i] > ema_100_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price below Kumo + TK Cross Bearish + 1W Downtrend
            elif (price < kumo_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  ema_100_1w_aligned[i] < ema_100_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Price crosses below Kumo midpoint
            if price < kumo_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Kumo midpoint
            if price > kumo_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_TK_1WTrend"
timeframe = "6h"
leverage = 1.0