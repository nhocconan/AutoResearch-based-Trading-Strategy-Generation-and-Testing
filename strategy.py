#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross with 1d Weekly Pivot Direction Filter
# Long when: price > Kumo (cloud), Tenkan > Kijun (TK cross bullish), AND price > 1d weekly pivot R1
# Short when: price < Kumo (cloud), Tenkan < Kijun (TK cross bearish), AND price < 1d weekly pivot S1
# Uses Ichimoku for trend/momentum, weekly pivots for institutional levels, discrete sizing (0.25) to minimize fee churn.
# Works in bull/bear via cloud filter (trend) + pivot levels (mean reversion at extremes).
# Timeframe: 6h (primary), HTF: 1d for weekly pivot calculation.

name = "6h_Ichimoku_TK_Cross_1dWeeklyPivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load HTF data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot points (using prior week's high/low/close)
    # We need to resample 1d to weekly - but since we can't resample, we approximate:
    # Use rolling 5-day window for weekly high/low/close (standard 5 trading days)
    if len(df_1d) >= 5:
        # Weekly high = max of last 5 daily highs
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
        # Weekly low = min of last 5 daily lows
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        # Weekly close = last daily close in the 5-day window
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).apply(lambda x: x[-1], raw=True).values
        
        # Calculate weekly pivot points: P = (H+L+C)/3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = 2*P - L
        weekly_r1 = 2 * weekly_pivot - weekly_low
        # S1 = 2*P - H
        weekly_s1 = 2 * weekly_pivot - weekly_high
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        # Not enough data for weekly calculation
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Current Kumo (cloud) boundaries: we need Senkou A and B from 26 periods ago
    # So current cloud is senkou_a[ i-26 ] and senkou_b[ i-26 ]
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to lag
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top = max(senkou_a, senkou_b), Kumo bottom = min(senkou_a, senkou_b)
    kumo_top = np.where(senkou_a_lagged > senkou_b_lagged, senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.where(senkou_a_lagged < senkou_b_lagged, senkou_a_lagged, senkou_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # warmup for Ichimoku (need 52 for Senkou B)
    
    for i in range(start_idx, n):
        # Skip if weekly pivot data not available
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_tenkan = tenkan[i]
        curr_kijun = kijun[i]
        curr_kumo_top = kumo_top[i]
        curr_kumo_bottom = kumo_bottom[i]
        curr_weekly_r1 = weekly_r1_aligned[i]
        curr_weekly_s1 = weekly_s1_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Kumo (cloud)
            # 2. Tenkan crosses below Kijun (TK cross bearish)
            # 3. Price falls below weekly S1 (mean reversion)
            if (curr_close < curr_kumo_bottom or
                curr_tenkan < curr_kijun or
                curr_close < curr_weekly_s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Kumo (cloud)
            # 2. Tenkan crosses above Kijun (TK cross bullish)
            # 3. Price rises above weekly R1 (mean reversion)
            if (curr_close > curr_kumo_top or
                curr_tenkan > curr_kijun or
                curr_close > curr_weekly_r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish Ichimoku: price > Kumo AND Tenkan > Kijun
            bullish_ichimoku = (curr_close > curr_kumo_top) and (curr_tenkan > curr_kijun)
            # Bearish Ichimoku: price < Kumo AND Tenkan < Kijun
            bearish_ichimoku = (curr_close < curr_kumo_bottom) and (curr_tenkan < curr_kijun)
            
            # Long entry: bullish Ichimoku AND price > weekly R1 (break above resistance)
            if bullish_ichimoku and (curr_close > curr_weekly_r1):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Ichimoku AND price < weekly S1 (break below support)
            elif bearish_ichimoku and (curr_close < curr_weekly_s1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals