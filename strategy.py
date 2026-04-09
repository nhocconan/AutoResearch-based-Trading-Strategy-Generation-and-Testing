#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with TK cross and volume confirmation
# Ichimoku Cloud from 1d timeframe provides major support/resistance and trend direction
# TK cross (Tenkan/Kijun) on 6h for entry timing with cloud filter from 1d
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters false signals
# Works in both bull and bear markets: cloud acts as dynamic support/resistance
# Position size fixed at 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_ichimoku_tk_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2.0
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Align 1d Ichimoku data to 6h timeframe (no additional delay needed for Senkou spans as they're already plotted ahead)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h TK cross (Tenkan/Kijun) for entry timing
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2.0
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2.0
    
    # TK cross signals
    tk_cross_above = tenkan_6h > kijun_6h  # Bullish cross
    tk_cross_below = tenkan_6h < kijun_6h  # Bearish cross
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine cloud boundaries and color
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bullish = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]  # Green cloud
        
        if position == 1:  # Long position
            # Exit on TK cross below OR price breaks below cloud bottom
            if tk_cross_below[i] or close[i] < cloud_bottom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on TK cross above OR price breaks above cloud top
            if tk_cross_above[i] or close[i] > cloud_top:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TK bullish cross AND price above cloud AND volume confirmed
            # Enter short: TK bearish cross AND price below cloud AND volume confirmed
            if volume_confirmed:
                if tk_cross_above[i] and close[i] > cloud_top:
                    position = 1
                    signals[i] = 0.25
                elif tk_cross_below[i] and close[i] < cloud_bottom:
                    position = -1
                    signals[i] = -0.25
    
    return signals