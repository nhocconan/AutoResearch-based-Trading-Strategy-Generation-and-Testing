#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1
Hypothesis: 6h Ichimoku cloud with TK cross entry, filtered by 1d trend (price vs 1d EMA50).
Long when Tenkan > Kijun and price above cloud, price > 1d EMA50.
Short when Tenkan < Kijun and price below cloud, price < 1d EMA50.
Uses cloud as dynamic support/resistance and TK cross for momentum.
Only takes trades in direction of 1d trend to avoid counter-trend whipsaws.
Designed for 12-37 trades/year by requiring confluence of TK cross, cloud position, and 1d trend.
Works in bull/bear via 1d trend filter: only takes longs in uptrend, shorts in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Load 1d data ONCE before loop for HTF trend and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku cloud: green when Senkou A > Senkou B, red when Senkou A < Senkou B
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]  # Bullish TK cross
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]  # Bearish TK cross
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Entry conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long when: bullish TK cross AND price above cloud
            if tk_cross_up and price_above_cloud:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long when: bearish TK cross OR price falls below cloud
            elif position == 1 and (tk_cross_down or price_below_cloud):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short when: bearish TK cross AND price below cloud
            if tk_cross_down and price_below_cloud:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short when: bullish TK cross OR price rises above cloud
            elif position == -1 and (tk_cross_up or price_above_cloud):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0