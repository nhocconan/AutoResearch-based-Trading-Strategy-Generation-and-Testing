#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_Filter
Hypothesis: On 6h timeframe, enter long when price breaks above Ichimoku cloud (Senkou Span A/B) with Kumo twist bullish (Senkou A > Senkou B) and 1w uptrend (close > 1w EMA50). Enter short when price breaks below cloud with Kumo twist bearish (Senkou A < Senkou B) and 1w downtrend. Uses discrete position size 0.25. Ichimoku cloud acts as dynamic support/resistance, Kumo twist indicates momentum shift, and 1w trend filter ensures alignment with higher timeframe bias. Designed for 12-30 trades/year by requiring multiple confluence factors, reducing false breakouts in choppy markets while capturing strong trending moves.
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
    
    # Get 1d data for Ichimoku calculation and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 50:  # Need enough for Ichimoku and EMA
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    # Senkou spans need no additional delay as they're already forward-shifted
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo twist: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
    kumo_twist_bullish = senkou_a_aligned > senkou_b_aligned
    kumo_twist_bearish = senkou_a_aligned < senkou_b_aligned
    
    # Cloud boundaries: upper cloud = max(Senkou A, Senkou B), lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52 periods) and 1w EMA warmup
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or np.isnan(kumo_twist_bullish[i]) or
            np.isnan(kumo_twist_bearish[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper cloud + Kumo twist bullish + 1w uptrend
            long_signal = (close[i] > upper_cloud[i]) and kumo_twist_bullish[i] and trend_1w_uptrend
            
            # Short: price breaks below lower cloud + Kumo twist bearish + 1w downtrend
            short_signal = (close[i] < lower_cloud[i]) and kumo_twist_bearish[i] and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below lower cloud OR Kumo twist turns bearish OR 1w trend turns down
            if (close[i] < lower_cloud[i] or not kumo_twist_bullish[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper cloud OR Kumo twist turns bullish OR 1w trend turns up
            if (close[i] > upper_cloud[i] or not kumo_twist_bearish[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0