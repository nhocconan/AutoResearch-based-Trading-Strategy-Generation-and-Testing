#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter
# Long when TK crosses above Kijun + price above Cloud (from 1d) + Tenkan > Kijun (bullish bias)
# Short when TK crosses below Kijun + price below Cloud + Tenkan < Kijun (bearish bias)
# Exit when TK crosses back across Kijun or price enters Cloud
# Uses 1-day Ichimoku for trend direction filter to avoid counter-trend trades
# Target: 60-120 total trades over 4 years (15-30/year)

name = "6h_ichimoku_1d_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1-day data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_10_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_10_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_10_9 + low_10_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_10_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_10_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_10_26 + low_10_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_10_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_10_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_10_52 + low_10_52) / 2
    
    # Cloud (Kumo): between Senkou A and B
    # For trend filter: price above cloud = bullish, price below cloud = bearish
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1-day Ichimoku to 6-hour
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # 6-hour Ichimoku for entry signals
    # Tenkan-sen (Conversion Line)
    high_6_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_6_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6 = (high_6_9 + low_6_9) / 2
    
    # Kijun-sen (Base Line)
    high_6_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_6_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6 = (high_6_26 + low_6_26) / 2
    
    # TK Cross signals
    tk_cross_above = (tenkan_6 > kijun_6) & (np.roll(tenkan_6, 1) <= np.roll(kijun_6, 1))
    tk_cross_below = (tenkan_6 < kijun_6) & (np.roll(tenkan_6, 1) >= np.roll(kijun_6, 1))
    
    # Handle first value
    tk_cross_above[0] = False
    tk_cross_below[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(26, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6[i]) or np.isnan(kijun_6[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: TK cross below or price enters cloud
            if tk_cross_below[i] or (close[i] >= cloud_bottom_1d_aligned[i] and close[i] <= cloud_top_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross above or price enters cloud
            if tk_cross_above[i] or (close[i] >= cloud_bottom_1d_aligned[i] and close[i] <= cloud_top_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with 1-day trend filter
            # Bullish: TK cross above + price above cloud + Tenkan > Kijun (1d)
            if (tk_cross_above[i] and 
                close[i] > cloud_top_1d_aligned[i] and 
                tenkan_1d_aligned[i] > kijun_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Bearish: TK cross below + price below cloud + Tenkan < Kijun (1d)
            elif (tk_cross_below[i] and 
                  close[i] < cloud_bottom_1d_aligned[i] and 
                  tenkan_1d_aligned[i] < kijun_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals