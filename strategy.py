#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly close for trend filter (SMA 20)
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used in signals to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(sma20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku cloud: Senkou A and B form the cloud
        # For bullish cloud: Senkou A > Senkou B
        # For bearish cloud: Senkou A < Senkou B
        bullish_cloud = senkou_a[i] > senkou_b[i]
        bearish_cloud = senkou_a[i] < senkou_b[i]
        
        # TK Cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long: TK cross up + bullish cloud + weekly trend up (price > weekly SMA20)
            if tk_cross_up and bullish_cloud and close[i] > sma20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + bearish cloud + weekly trend down (price < weekly SMA20)
            elif tk_cross_down and bearish_cloud and close[i] < sma20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross down OR price below cloud (Senkou B)
            if tk_cross_down or close[i] < senkou_b[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross up OR price above cloud (Senkou A)
            if tk_cross_up or close[i] > senkou_a[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals