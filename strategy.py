#!/usr/bin/env python3
# 6h_Ichimoku_1dTrend_VolumeFilter
# Hypothesis: Uses Ichimoku cloud from daily timeframe for trend direction and entry timing.
# Enters long when Tenkan-sen crosses above Kijun-sen and price is above the cloud (bullish).
# Enters short when Tenkan-sen crosses below Kijun-sen and price is below the cloud (bearish).
# Uses volume confirmation (current volume > 1.5 * 20-period average) to avoid false signals.
# Exits when price crosses the opposite Ichimoku line (Tenkan-sen crosses back).
# Ichimoku works well in trending markets and the cloud acts as dynamic support/resistance.
# The daily timeframe provides a robust trend filter suitable for both bull and bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6h_Ichimoku_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Warmup for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below the cloud
        # Cloud top is the higher of Senkou Span A and B
        # Cloud bottom is the lower of Senkou Span A and B
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan-sen / Kijun-sen cross
        tk_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long entry: TK cross above + price above cloud + volume confirmation
            if tk_cross_above and price_above_cloud and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross below + price below cloud + volume confirmation
            elif tk_cross_below and price_below_cloud and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross back below (regardless of cloud)
            if tk_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross back above (regardless of cloud)
            if tk_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals