#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1dTrend_v1
Hypothesis: Ichimoku cloud on 6h with 1d EMA50 trend filter to capture strong trend continuation in both bull and bear markets. Uses Tenkan/Kijun cross for entry timing and cloud for trend direction filter. Only trades when 6h price is above/below cloud AND aligned with 1d trend. Targets 12-30 trades/year via tight entry requiring trend alignment and momentum confirmation.
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
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Using 1d for simplicity; could use 6h Ichimoku
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) + 1d EMA(50)
    start_idx = max(52, 50) + 26  # +26 for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Ichimoku signals
        # Bullish: Tenkan crosses above Kijun AND price above cloud
        bullish_cross = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1] if i > 0 else False
        price_above_cloud = close_val > cloud_top
        
        # Bearish: Tenkan crosses below Kijun AND price below cloud
        bearish_cross = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1] if i > 0 else False
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Long: bullish cross above cloud with uptrend
            if bullish_cross and price_above_cloud and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish cross below cloud with downtrend
            elif bearish_cross and price_below_cloud and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below cloud or Tenkan crosses below Kijun
            if close_val < cloud_bottom or (tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above cloud or Tenkan crosses above Kijun
            if close_val > cloud_top or (tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrend_v1"
timeframe = "6h"
leverage = 1.0