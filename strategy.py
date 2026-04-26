#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirmation
Hypothesis: Ichimoku cloud breakout on 6h with 12h trend filter (price above/below Kumo twist) and volume confirmation (>1.4x 20-period MA). 
Long when price breaks above Kumo cloud in 12h uptrend with volume spike. Short when price breaks below Kumo cloud in 12h downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. 
Ichimoku components calculated on 6h timeframe, trend filter from 12h timeframe.
Designed to capture strong trending moves while avoiding whipsaws in ranging markets via cloud filter and volume confirmation.
Target: 12-37 trades/year (50-150 total over 4 years).
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Ichimoku components on 6h timeframe
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((high_senkou + low_senkou) / 2)
    
    # Lagging Span (Chikou Span): Close plotted 26 periods behind
    # Not used for entry/exit but confirms trend strength
    
    # Align Ichimoku components (no look-ahead: align_htf_to_ltf handles completed bar timing)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no shift needed
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    # Kumo cloud boundaries (Senkou Span A and B)
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 12h trend filter: price above/below Kumo twist (Senkou Span A/B crossover)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Kumo on 12h timeframe
    high_tenkan_12h = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan_12h = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_12h = (high_tenkan_12h + low_tenkan_12h) / 2
    
    high_kijun_12h = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun_12h = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_12h = (high_kijun_12h + low_kijun_12h) / 2
    
    senkou_span_a_12h = ((tenkan_12h + kijun_12h) / 2)
    high_senkou_12h = pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_12h = pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_12h = ((high_senkou_12h + low_senkou_12h) / 2)
    
    # Kumo twist: Senkou Span A crossing above/below Senkou Span B
    # Uptrend when Senkou Span A > Senkou Span B (bullish twist)
    # Downtrend when Senkou Span A < Senkou Span B (bearish twist)
    kumo_twist_12h = senkou_span_a_12h - senkou_span_b_12h
    uptrend_12h = kumo_twist_12h > 0
    downtrend_12h = kumo_twist_12h < 0
    
    # Align 12h trend filter to 6h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.4x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B + 26 for displacement + 20 for volume MA)
    start_idx = 98
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper cloud with 12h uptrend and volume spike
            if (close[i] > upper_cloud[i] and 
                uptrend_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower cloud with 12h downtrend and volume spike
            elif (close[i] < lower_cloud[i] and 
                  downtrend_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below lower cloud (cloud break) OR 12h trend changes to downtrend
            if (close[i] < lower_cloud[i] or not uptrend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above upper cloud (cloud break) OR 12h trend changes to uptrend
            if (close[i] > upper_cloud[i] or not downtrend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0