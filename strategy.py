#!/usr/bin/env python3
# 6h_ichimoku_1d_cloud_filter_v3
# Hypothesis: 6h strategy using 1d Ichimoku cloud as trend filter and TK cross for entries.
# Long: Price above cloud, Tenkan crosses above Kijun, and volume > 1.5x 20-period average.
# Short: Price below cloud, Tenkan crosses below Kijun, and volume > 1.5x 20-period average.
# Exit: Price crosses opposite cloud boundary (top for longs, bottom for shorts).
# Uses 1d Ichimoku for higher timeframe trend alignment to work in both bull and bear markets.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_cloud_filter_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals but calculated for completeness
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # TK cross: Tenkan crosses Kijun
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: Price crosses below cloud bottom
            if close[i] < cloud_bottom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above cloud top
            if close[i] > cloud_top:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud, TK cross up, and volume confirmation
            if (close[i] > cloud_top and           # Price above cloud
                tk_cross_up and                    # Tenkan crosses above Kijun
                volume_confirmed):                 # Volume spike
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud, TK cross down, and volume confirmation
            elif (close[i] < cloud_bottom and      # Price below cloud
                  tk_cross_down and                # Tenkan crosses below Kijun
                  volume_confirmed):               # Volume spike
                position = -1
                signals[i] = -0.25
    
    return signals