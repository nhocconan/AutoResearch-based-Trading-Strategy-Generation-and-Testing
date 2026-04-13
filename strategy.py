#!/usr/bin/env python3
"""
1d_1w_Ichimoku_Bullish_Trend_Following
Hypothesis: Ichimoku Cloud on daily chart with weekly trend filter captures major trends.
Price above/below Cloud + weekly Senkou Span filter gives high-probability trend trades.
Works in bull (trend following) and bear (short when price below Cloud + weekly filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).mean()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).mean()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).mean()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).mean()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).mean()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).mean()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Get weekly Senkou Span B for trend filter
    high_52_w = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).mean()
    low_52_w = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).mean()
    senkou_b_w = ((high_52_w + low_52_w) / 2).values
    senkou_b_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(52, n):
        # Skip if any required data is not ready
        if (np.isnan(tenkan.iloc[i]) or np.isnan(kijun.iloc[i]) or 
            np.isnan(senkou_a.iloc[i]) or np.isnan(senkou_b.iloc[i]) or
            np.isnan(senkou_b_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a.iloc[i], senkou_b.iloc[i])
        cloud_bottom = min(senkou_a.iloc[i], senkou_b.iloc[i])
        
        # Weekly trend filter: price relative to weekly Senkou B
        weekly_uptrend = close[i] > senkou_b_w_aligned[i]
        weekly_downtrend = close[i] < senkou_b_w_aligned[i]
        
        # Long signal: price above Cloud + weekly uptrend
        long_signal = (close[i] > cloud_top) and weekly_uptrend
        
        # Short signal: price below Cloud + weekly downtrend
        short_signal = (close[i] < cloud_bottom) and weekly_downtrend
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Ichimoku_Bullish_Trend_Following"
timeframe = "1d"
leverage = 1.0