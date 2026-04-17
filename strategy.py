#!/usr/bin/env python3
"""
12h_Ichimoku_Cloud_Trend_Strategy
Strategy: 12h Ichimoku Cloud with Tenkan/Kijun cross and price outside cloud.
Long: Tenkan > Kijun AND price above Senkou Span A/B (above cloud)
Short: Tenkan < Kijun AND price below Senkou Span A/B (below cloud)
Exit: Tenkan/Kijun cross reverses OR price enters cloud
Position size: 0.25
Trend-following strategy designed to capture sustained moves while avoiding whipsaws in ranging markets.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals to avoid look-ahead
    
    # Daily trend filter from 1D timeframe
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period volume average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need Senkou B calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        # Price position relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_dema = close[i] > ema50_1d_aligned[i]
        price_below_dema = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Tenkan > Kijun AND price above cloud + volume + trend filter
            if (tenkan_above_kijun and price_above_cloud and 
                volume_filter and price_above_dema):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below cloud + volume + trend filter
            elif (tenkan_below_kijun and price_below_cloud and 
                  volume_filter and price_below_dema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan/Kijun cross down OR price enters cloud
            if (tenkan_below_kijun or price_in_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan/Kijun cross up OR price enters cloud
            if (tenkan_above_kijun or price_in_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Ichimoku_Cloud_Trend_Strategy"
timeframe = "12h"
leverage = 1.0