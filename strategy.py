#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (Tenkan/Kijun cross + price above/below cloud).
Long when price breaks above cloud in 1d uptrend with Tenkan > Kijun. Short when price breaks below cloud in 1d downtrend with Tenkan < Kijun.
Uses discrete position sizing (0.25) to minimize fee churn. Ichimoku calculated with proper lookback to avoid look-ahead.
Designed to work in both bull and bear markets by following the 1d trend. Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().shift(1)
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().shift(1)
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().shift(1)
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().shift(1)
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().shift(1)
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().shift(1)
    senkou_b = ((high_52 + low_52) / 2)
    
    # The actual cloud Senkou Span A/B are plotted 26 periods ahead,
    # so to get current cloud we need values shifted BACK by 26
    senkou_a_lagged = pd.Series(senkou_a).shift(26).values
    senkou_b_lagged = pd.Series(senkou_b).shift(26).values
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 52 for Senkou B + 26 for shift + 1 for Ichimoku shift + 50 for 1d EMA
    start_idx = 129
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        upper_cloud = np.maximum(senkou_a_lagged[i], senkou_b_lagged[i])
        lower_cloud = np.minimum(senkou_a_lagged[i], senkou_b_lagged[i])
        
        if position == 0:
            # Long: price breaks above cloud with 1d uptrend and bullish TK cross (Tenkan > Kijun)
            if (close[i] > upper_cloud and 
                uptrend_1d[i] and tenkan[i] > kijun[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with 1d downtrend and bearish TK cross (Tenkan < Kijun)
            elif (close[i] < lower_cloud and 
                  downtrend_1d[i] and tenkan[i] < kijun[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below cloud OR 1d trend changes to downtrend OR TK cross turns bearish
            if (close[i] < lower_cloud or not uptrend_1d[i] or tenkan[i] <= kijun[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above cloud OR 1d trend changes to uptrend OR TK cross turns bullish
            if (close[i] > upper_cloud or not downtrend_1d[i] or tenkan[i] >= kijun[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0