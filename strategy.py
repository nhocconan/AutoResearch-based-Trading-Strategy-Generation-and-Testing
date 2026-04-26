#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v1
Hypothesis: On 6h timeframe, trade Ichimoku cloud twists (Senkou Span A/B cross) with 1d trend filter (price vs EMA50) and volume confirmation. Ichimoku cloud twist indicates momentum shift, 1d EMA50 ensures alignment with higher timeframe trend, volume confirms participation. Designed for 6h to capture medium-term swings in both bull and bear markets by following the 1d trend while using 6h Ichimoku for timely entries and exits.
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
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)  # Use 1d data for alignment
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and 1d EMA (50)
    start_idx = max(52, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen[i]) or
            np.isnan(kijun_sen[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        vol_val = volume[i]
        avg_vol = pd.Series(volume).rolling(window=24, min_periods=24).mean().iloc[i] if i >= 24 else 0
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        
        # Volume confirmation: above average volume
        volume_filter = vol_val > avg_vol if avg_vol > 0 else True
        
        # Kumo twist detection: Senkou Span A/B cross
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        if i > 0:
            prev_senkou_a = senkou_span_a_aligned[i-1]
            prev_senkou_b = senkou_span_b_aligned[i-1]
            bullish_twist = (senkou_a > senkou_b) and (prev_senkou_a <= prev_senkou_b)
            bearish_twist = (senkou_a < senkou_b) and (prev_senkou_a >= prev_senkou_b)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        if position == 0:
            # Long: bullish Kumo twist with uptrend and volume
            long_signal = bullish_twist and uptrend and volume_filter
            
            # Short: bearish Kumo twist with downtrend and volume
            short_signal = bearish_twist and downtrend and volume_filter
            
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
            # Exit: bearish Kumo twist or price below Kumo (cloud)
            if bearish_twist or close_val < min(senkou_a, senkou_b):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish Kumo twist or price above Kumo (cloud)
            if bullish_twist or close_val > max(senkou_a, senkou_b):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v1"
timeframe = "6h"
leverage = 1.0