#!/usr/bin/env python3
"""
1d_Ichimoku_TK_Cross_WeeklyTrend
Hypothesis: Ichimoku Tenkan-Kijun cross on daily with weekly trend filter (price above/below weekly Kumo) captures major trend changes while avoiding whipsaws. Works in bull/bear via trend filter and avoids chop via Kumo twist. Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
"""

name = "1d_Ichimoku_TK_Cross_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 52 days for weekly calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo (Cloud) top and bottom
    kumotop = np.maximum(senkou_a, senkou_b)
    kumobottom = np.minimum(senkou_a, senkou_b)
    
    # Align weekly Kumo to daily (need to shift forward 26 periods for lookahead avoidance)
    # Senkou Span A and B are plotted 26 periods ahead
    senkou_a_weekly = pd.Series(df_weekly['high']).rolling(window=52, min_periods=52).max().values
    senkou_a_weekly = (senkou_a_weekly + pd.Series(df_weekly['low']).rolling(window=52, min_periods=52).min().values) / 2
    senkou_b_weekly = pd.Series(df_weekly['high']).rolling(window=26, min_periods=26).max().values
    senkou_b_weekly = (senkou_b_weekly + pd.Series(df_weekly['low']).rolling(window=26, min_periods=26).min().values) / 2
    
    # Weekly Kumo components (already forward-shifted by Ichimoku calculation)
    wk_kumotop = np.maximum(senkou_a_weekly, senkou_b_weekly)
    wk_kumobottom = np.minimum(senkou_a_weekly, senkou_b_weekly)
    
    # Align weekly Kumo to daily
    wk_kumotop_aligned = align_htf_to_ltf(prices, df_weekly, wk_kumotop)
    wk_kumobottom_aligned = align_htf_to_ltf(prices, df_weekly, wk_kumobottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above weekly Kumo
            if (tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i] and 
                close[i] > wk_kumotop_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below weekly Kumo
            elif (tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i] and 
                  close[i] < wk_kumobottom_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price closes below weekly Kumo bottom
            if (tenkan[i-1] > kijun[i-1] and tenkan[i] <= kijun[i]) or \
               close[i] < wk_kumobottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price closes above weekly Kumo top
            if (tenkan[i-1] < kijun[i-1] and tenkan[i] >= kijun[i]) or \
               close[i] > wk_kumotop_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals