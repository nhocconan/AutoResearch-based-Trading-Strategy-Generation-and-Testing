#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: On 6h timeframe, enter long when price breaks above Kumo (cloud) AND Kumo is bullish (Senkou Span A > Senkou Span B) AND 1d trend is up (close > EMA50). Enter short when price breaks below Kumo AND Kumo is bearish (Senkou Span A < Senkou Span B) AND 1d trend is down (close < EMA50). Exit on Kumo twist (Senkou Span A crosses Senkou Span B) or price retracing to Kumo midpoint. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 12-37 trades/year.
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
    
    # Get 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals but needed for alignment
    
    # Kumo (cloud) boundaries: Senkou Span A and Senkou Span B
    # Kumo is bullish when Senkou Span A > Senkou Span B
    # Kumo is bearish when Senkou Span A < Senkou Span B
    
    # Kumo twist: when Senkou Span A crosses Senkou Span B
    # We'll detect this as a change in the relationship
    
    # Kumo midpoint: (Senkou Span A + Senkou Span B)/2
    kumomid = (senkou_span_a + senkou_span_b) / 2
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all Ichimoku components and EMA50 to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    kumomid_aligned = align_htf_to_ltf(prices, df_1d, kumomid)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52 periods) and EMA warmup (50 periods)
    start_idx = max(52, 50)  # Ichimoku needs 52, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(kumomid_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Kumo boundaries
        upper_kumo = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Kumo is bullish when Senkou Span A > Senkou Span B
        kumo_bullish = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        kumo_bearish = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Kumo twist detection: change in Senkou Span A/B relationship
        # We'll use the previous bar to detect twist
        if i > start_idx:
            prev_kumo_bullish = senkou_span_a_aligned[i-1] > senkou_span_b_aligned[i-1]
            prev_kumo_bearish = senkou_span_a_aligned[i-1] < senkou_span_b_aligned[i-1]
            kumo_twist = (prev_kumo_bullish and not kumo_bullish) or (prev_kumo_bearish and not kumo_bearish)
        else:
            kumo_twist = False
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Kumo AND Kumo is bullish AND 1d uptrend
            long_signal = close[i] > upper_kumo and kumo_bullish and trend_uptrend
            
            # Short: price breaks below lower Kumo AND Kumo is bearish AND 1d downtrend
            short_signal = close[i] < lower_kumo and kumo_bearish and trend_downtrend
            
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
            # Exit: Kumo twist (bullish to bearish) OR price retracing to Kumo midpoint
            if kumo_twist or close[i] < kumomid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Kumo twist (bearish to bullish) OR price retracing to Kumo midpoint
            if kumo_twist or close[i] > kumomid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0