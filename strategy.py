#!/usr/bin/env python3
# 6h_ichimoku_cloud_regime_v1
# Hypothesis: 6h strategy using daily Ichimoku cloud for regime detection and 6h Tenkan-Kijun cross for entry.
# Long when price > daily cloud (bullish regime) AND 6h Tenkan crosses above Kijun.
# Short when price < daily cloud (bearish regime) AND 6h Tenkan crosses below Kijun.
# Exit when price crosses back into daily cloud or Tenkan-Kijun cross reverses.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Ichimoku cloud acts as dynamic support/resistance and trend filter, effective in both bull and bear markets.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Calculate daily Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Tenkan and Kijun for entry signals
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Determine if price is above or below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Calculate 6h Tenkan-Kijun cross
        tk_cross_6h = tenkan_6h[i] - kijun_6h[i]
        tk_cross_6h_prev = tenkan_6h[i-1] - kijun_6h[i-1] if i > 0 else 0
        
        bullish_tk_cross = tk_cross_6h > 0 and tk_cross_6h_prev <= 0
        bearish_tk_cross = tk_cross_6h < 0 and tk_cross_6h_prev >= 0
        
        if position == 1:  # Long position
            # Exit: Price crosses back into cloud OR Tenkan-Kijun cross reverses to bearish
            if (not price_above_cloud) or bearish_tk_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses back into cloud OR Tenkan-Kijun cross reverses to bullish
            if (not price_below_cloud) or bullish_tk_cross:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry: price outside cloud AND Tenkan-Kijun cross in same direction
            bullish_entry = price_above_cloud and bullish_tk_cross
            bearish_entry = price_below_cloud and bearish_tk_cross
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals