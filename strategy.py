#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter. Uses Ichimoku (Tenkan/Kijun/Senkou A/B) on 6h for entry/exit and 1d EMA50 for trend filter.
Long when price > cloud, Tenkan > Kijun, and 1d EMA50 rising. Short when price < cloud, Tenkan < Kijun, and 1d EMA50 falling.
Exit when price crosses opposite cloud boundary or Tenkan/Kijun cross reverses.
Ichimoku provides dynamic support/resistance; 1d EMA50 filter avoids counter-trend whipsaws in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year). Works in bull markets (captures uptrends via cloud) and bear markets (captures downtrends via cloud).
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components on 6h timeframe
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    close_6h_series = pd.Series(close_6h)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = high_6h_series.rolling(window=9, min_periods=9).max()
    period9_low = low_6h_series.rolling(window=9, min_periods=9).min()
    tenkan = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = high_6h_series.rolling(window=26, min_periods=26).max()
    period26_low = low_6h_series.rolling(window=26, min_periods=26).min()
    kijun = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # We'll handle the shift in alignment - no need to shift here as align_htf_to_ltf handles timing
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period52_high = high_6h_series.rolling(window=52, min_periods=52).max()
    period52_low = low_6h_series.rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h Ichimoku components to 6h timeframe (no alignment needed for same TF)
    tenkan_aligned = tenkan
    kijun_aligned = kijun
    senkou_a_aligned = senkou_a
    senkou_b_aligned = senkou_b
    
    # Align 1d EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA50 slope for trend direction (1-bar change)
    ema50_slope = np.zeros_like(ema50_aligned)
    ema50_slope[1:] = ema50_aligned[1:] - ema50_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema50_slope[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema50_val = ema50_aligned[i]
        ema50_slope_val = ema50_slope[i]
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: price > cloud, Tenkan > Kijun, and EMA50 sloping up
            if price > upper_cloud and tenkan_val > kijun_val and ema50_slope_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price < cloud, Tenkan < Kijun, and EMA50 sloping down
            elif price < lower_cloud and tenkan_val < kijun_val and ema50_slope_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < lower cloud OR Tenkan < Kijun (trend weakening)
            if price < lower_cloud or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > upper cloud OR Tenkan > Kijun (trend reversing)
            if price > upper_cloud or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_Trend_Filter"
timeframe = "6h"
leverage = 1.0