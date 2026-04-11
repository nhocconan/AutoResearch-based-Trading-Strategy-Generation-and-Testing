#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Ichimoku cloud and daily Tenkan/Kijun cross.
# Uses weekly Ichimoku for trend direction (price above/below cloud) and daily Tenkan/Kijun cross for entry timing.
# Long when price above weekly cloud and daily Tenkan crosses above Kijun.
# Short when price below weekly cloud and daily Tenkan crosses below Kijun.
# Designed for low trade frequency (~15-30/year) to minimize fee decay while capturing medium-term trends.
# Works in bull/bear markets by using cloud as dynamic support/resistance and TK cross for momentum confirmation.

name = "6h_1w_1d_ichimoku_tk_cross_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly Ichimoku cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2)
    
    # The cloud is between Senkou Span A and B
    # For simplicity, we use the average of Senkou Span A and B as cloud center,
    # and half the difference as cloud thickness
    # But for trend direction, price > max(Span A, Span B) = above cloud
    # price < min(Span A, Span B) = below cloud
    span_a_1w = senkou_span_a_1w
    span_b_1w = senkou_span_b_1w
    
    # Align weekly Ichimoku components to 6h timeframe
    span_a_aligned = align_htf_to_ltf(prices, df_1w, span_a_1w)
    span_b_aligned = align_htf_to_ltf(prices, df_1w, span_b_1w)
    # For cloud center and thickness, we'll use the aligned spans directly
    
    # Daily Tenkan/Kijun for entry signal
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan-sen (9-period)
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Daily Kijun-sen (26-period)
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Align daily Tenkan/Kijun to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    # Calculate crossover signals
    # Tenkan crosses above Kijun: bullish
    tk_cross_up = (tenkan_aligned > kijun_aligned) & (np.roll(tenkan_aligned, 1) <= np.roll(kijun_aligned, 1))
    # Tenkan crosses below Kijun: bearish
    tk_cross_down = (tenkan_aligned < kijun_aligned) & (np.roll(tenkan_aligned, 1) >= np.roll(kijun_aligned, 1))
    # Handle first element
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 26 to ensure all indicators are valid
    for i in range(26, n):
        # Skip if any required data is invalid
        if (np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine if price is above or below weekly cloud
        above_cloud = (close[i] > span_a_aligned[i]) and (close[i] > span_b_aligned[i])
        below_cloud = (close[i] < span_a_aligned[i]) and (close[i] < span_b_aligned[i])
        
        # Entry conditions
        long_entry = above_cloud and tk_cross_up[i]
        short_entry = below_cloud and tk_cross_down[i]
        
        # Exit conditions: reverse signal or price enters cloud
        exit_long = below_cloud or tk_cross_down[i]
        exit_special = above_cloud or tk_cross_up[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_special:  # Exit short when price goes above cloud or bullish cross
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals