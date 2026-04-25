#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_v1
Hypothesis: Trade Ichimoku cloud breakouts on 6h with 1w trend filter. In bull markets (price > weekly Kumo), go long on Tenkan-Kijun cross above cloud. In bear markets (price < weekly Kumo), go short on Tenkan-Kijun cross below cloud. Uses weekly trend to avoid counter-trend whipsaws. Target: 15-30 trades/year on BTC/ETH to stay under 300-trade 6h hard max. Works in bull (cloud breakouts with uptrend) and bear (cloud breakdowns with downtrend).
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
    
    # Get 1w data for HTF trend filter (weekly Kumo/cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen_1w = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen_1w = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_1w = (max_high_52 + min_low_52) / 2
    
    # Align weekly Ichimoku components to 6h timeframe (wait for weekly bar close)
    tenkan_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_1w)
    kijun_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # Calculate Kumo (cloud) boundaries: Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    kumo_top_1w = np.maximum(senkou_span_a_1w_aligned, senkou_span_b_1w_aligned)
    kumo_bottom_1w = np.minimum(senkou_span_a_1w_aligned, senkou_span_b_1w_aligned)
    
    # Calculate 6h Ichimoku for entry signals (Tenkan/Kijun cross)
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_9_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_9_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_sen_6h = (max_high_9_6h + min_low_9_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_sen_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK cross signals
    tk_cross_above = (tenkan_sen_6h > kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) <= np.roll(kijun_sen_6h, 1))
    tk_cross_below = (tenkan_sen_6h < kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) >= np.roll(kijun_sen_6h, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 6h indicators (26) and weekly alignment
    start_idx = max(26, 52)  # 26 for 6h Kijun, 52 for weekly Senkou B
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if (np.isnan(kumo_top_1w[i]) or np.isnan(kumo_bottom_1w[i]) or 
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: price above/below weekly Kumo
        price_above_kumo = close[i] > kumo_top_1w[i]
        price_below_kumo = close[i] < kumo_bottom_1w[i]
        
        if position == 0:
            # Long setup: TK cross above + price above weekly Kumo (bullish regime)
            long_setup = tk_cross_above[i] and price_above_kumo
            
            # Short setup: TK cross below + price below weekly Kumo (bearish regime)
            short_setup = tk_cross_below[i] and price_below_kumo
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK cross below OR price falls below weekly Kumo (trend change)
            if tk_cross_below[i] or (close[i] < kumo_top_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK cross above OR price rises above weekly Kumo (trend change)
            if tk_cross_above[i] or (close[i] > kumo_bottom_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_v1"
timeframe = "6h"
leverage = 1.0