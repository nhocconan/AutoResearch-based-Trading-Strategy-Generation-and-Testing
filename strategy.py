#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) for trend direction.
- Entry: Long when price breaks above 6h Kumo (cloud) AND price > 1d Kijun-sen (bullish bias) AND volume > 1.5 * volume MA(20).
         Short when price breaks below 6h Kumo AND price < 1d Kijun-sen (bearish bias) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price closes below 6h Kumo,
        exit short when price closes above 6h Kumo.
- Signal size: 0.25 discrete to balance return and drawdown.
Ichimoku provides dynamic support/resistance and trend identification, effective in both trending and ranging markets.
The 1d Kijun-sen acts as a higher-timeframe trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_10 + min_low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align HTF Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Ichimoku components for cloud (Kumo)
    period_tenkan_6h = 9
    period_kijun_6h = 26
    period_senkou_b_6h = 52
    
    max_high_10_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_10_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_sen_6h = (max_high_10_6h + min_low_10_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_sen_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    senkou_span_a_6h = (tenkan_sen_6h + kijun_sen_6h) / 2
    
    max_high_52_6h = pd.Series(high).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).max().values
    min_low_52_6h = pd.Series(low).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).min().values
    senkou_span_b_6h = (max_high_52_6h + min_low_52_6h) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For cloud top/bottom, we need to shift Senkou Span by 26 periods ahead
    # But for breakout detection, we use current cloud (already plotted)
    # The cloud is between senkou_span_a and senkou_span_b
    # We'll use the current values for breakout detection
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need enough for 1d Ichimoku (52 periods) and 6h Ichimoku (52 periods)
    start_idx = max(100, 60)  # Conservative start
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Determine cloud boundaries for 6h
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above 6h cloud AND price > 1d Kijun-sen (bullish bias) AND volume confirmed
            if curr_close > cloud_top and curr_close > kijun_sen_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h cloud AND price < 1d Kijun-sen (bearish bias) AND volume confirmed
            elif curr_close < cloud_bottom and curr_close < kijun_sen_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below 6h cloud (trend change)
            if curr_close < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above 6h cloud (trend change)
            if curr_close > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dKijunTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0