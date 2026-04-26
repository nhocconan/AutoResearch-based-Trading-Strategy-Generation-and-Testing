#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend
Hypothesis: Use 6h timeframe with Ichimoku TK cross (Tenkan/Kijun) confirmed by price position relative to Kumo (cloud) and 1w trend filter.
Long when: TK cross bullish (Tenkan > Kijun) + price above Kumo (Senkou Span A & B) + 1w EMA50 uptrend.
Short when: TK cross bearish (Tenkan < Kijun) + price below Kumo + 1w EMA50 downtrend.
Exit when: TK cross reverses or price crosses Kumo in opposite direction.
Uses discrete 0.25 position size. Targets 12-30 trades/year to avoid fee drag.
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    highest_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_b = ((highest_52 + lowest_52) / 2)
    
    # For cloud calculation, we need to shift Senkou A and B forward by displacement
    # But for signal generation at time t, we use Senkou A and B from t-displacement
    # So we align by using values that were calculated displacement periods ago
    senkou_a_lagged = np.roll(senkou_a, displacement)
    senkou_b_lagged = np.roll(senkou_b, displacement)
    # First 'displacement' values are invalid due to roll
    senkou_a_lagged[:displacement] = np.nan
    senkou_b_lagged[:displacement] = np.nan
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need max of all periods + displacement
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + displacement
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        # Determine Kumo (cloud) boundaries
        upper_cloud = max(senkou_a_lagged[i], senkou_b_lagged[i])
        lower_cloud = min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        if position == 0:
            # Flat - look for TK cross with cloud and trend confirmation
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            
            # Price above cloud (bullish) or below cloud (bearish)
            price_above_cloud = close_val > upper_cloud
            price_below_cloud = close_val < lower_cloud
            
            # 1w trend: EMA50 rising or falling
            ema_rising = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            ema_falling = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
            
            # Long: bullish TK cross + price above cloud + 1w uptrend
            long_entry = tk_bullish_cross and price_above_cloud and ema_rising
            # Short: bearish TK cross + price below cloud + 1w downtrend
            short_entry = tk_bearish_cross and price_below_cloud and ema_falling
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross turns bearish or price drops below cloud
            tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            price_below_cloud = close_val < lower_cloud
            
            if tk_bearish_cross or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross turns bullish or price rises above cloud
            tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            price_above_cloud = close_val > upper_cloud
            
            if tk_bullish_cross or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend"
timeframe = "6h"
leverage = 1.0