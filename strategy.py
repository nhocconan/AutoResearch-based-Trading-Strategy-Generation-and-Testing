#!/usr/bin/env python3
"""
6h_1d_1w_Camarilla_Pivot_With_Trend_Filter
Hypothesis: Use daily Camarilla pivot levels (H3/L3) as support/resistance with 
weekly trend filter to avoid counter-trend trades. Enter on bounce from H3/L3 
with volume confirmation only when aligned with weekly trend. Weekly trend 
determined by price vs weekly 50 EMA. This avoids false breakouts in ranging 
markets and captures trend continuations in trending markets. Designed for 
low frequency (~15-30/year) with high win rate by requiring trend alignment.
Works in bull markets (buy H3 bounce in uptrend) and bear markets (sell L3 
bounce in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Camarilla_Pivot_With_Trend_Filter"
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
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === WEEKLY TREND FILTER: 50 EMA ON WEEKLY CHART ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: bounce from H3 as support in weekly uptrend
        long_signal = (weekly_uptrend and 
                      low[i] <= h3_aligned[i] and  # Touched or went below H3
                      close[i] > h3_aligned[i] and   # Closed back above H3 (bounce)
                      abs(close[i] - h3_aligned[i]) / h3_aligned[i] < 0.01 and  # Within 1% of H3
                      strong_volume)
        
        # Short: bounce from L3 as resistance in weekly downtrend
        short_signal = (weekly_downtrend and 
                       high[i] >= l3_aligned[i] and  # Touched or went above L3
                       close[i] < l3_aligned[i] and   # Closed back below L3 (bounce)
                       abs(close[i] - l3_aligned[i]) / l3_aligned[i] < 0.01 and  # Within 1% of L3
                       strong_volume)
        
        # Exit: opposite H3/L3 level or weekly trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_aligned[i] or not weekly_uptrend))
        exit_short = (position == -1 and 
                     (close[i] > h3_aligned[i] or not weekly_downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals