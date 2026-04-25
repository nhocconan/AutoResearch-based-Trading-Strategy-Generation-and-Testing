#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_Confluence
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with 1-day cloud filter and session timing.
Targets 12-25 trades/year by requiring: 1) TK cross on 6h (Tenkan(9) crosses Kijun(26)), 
2) Price above/below 1-day Ichimoku cloud for trend alignment, 3) UTC 08-20 session for liquidity.
The Ichimoku cloud acts as dynamic support/resistance and trend filter, reducing false signals.
Works in bull/bear markets by following the cloud's trend direction.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 6h Ichimoku components: Tenkan(9), Kijun(26)
    # Tenkan-sen = (HH(9) + LL(9)) / 2
    hh9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    ll9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (hh9 + ll9) / 2
    
    # Kijun-sen = (HH(26) + LL(26)) / 2
    hh26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    ll26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (hh26 + ll26) / 2
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components: Tenkan(9), Kijun(26), Senkou Span A/B(26)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen = (HH(9) + LL(9)) / 2
    hh9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    ll9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (hh9_1d + ll9_1d) / 2
    
    # 1d Kijun-sen = (HH(26) + LL(26)) / 2
    hh26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    ll26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (hh26_1d + ll26_1d) / 2
    
    # 1d Senkou Span A = (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B = (HH(52) + LL(52)) / 2, plotted 26 periods ahead
    hh52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    ll52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (hh52_1d + ll52_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud boundaries: max/min of Senkou Span A/B
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 6h Kijun(26) + 1d Senkou(52) + alignment
    start_idx = max(26, 52) + 26  # Conservative warmup for alignment
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # TK cross signals
        tk_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to cloud
        price_above_cloud = curr_close > upper_cloud[i]
        price_below_cloud = curr_close < lower_cloud[i]
        
        if position == 0:
            # Look for entry signals with cloud filter
            # Long: TK bullish cross + price above cloud
            if tk_bullish and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross + price below cloud
            elif tk_bearish and price_below_cloud:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on TK bearish cross or price drops below cloud
            if tk_bearish or curr_close < lower_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on TK bullish cross or price rises above cloud
            if tk_bullish or curr_close > upper_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_Confluence"
timeframe = "6h"
leverage = 1.0