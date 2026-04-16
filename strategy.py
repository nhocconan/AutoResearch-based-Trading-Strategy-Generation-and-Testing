#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Tenkan-Kijun cross and weekly Kumo filter.
# Long when price > Kumo AND Tenkan > Kijun (bullish momentum) AND weekly Kumo is bullish (Senkou Span A > Senkou Span B).
# Short when price < Kumo AND Tenkan < Kijun (bearish momentum) AND weekly Kumo is bearish (Senkou Span A < Senkou Span B).
# Uses discrete position size 0.25. Ichimoku provides trend, momentum, and support/resistance in one system.
# Weekly Kumo filter ensures alignment with higher timeframe trend structure.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: Ichimoku Components ===
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
    
    # Current Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # But for price vs cloud comparison, we use current Senkou spans
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Get 1w data once before loop for weekly Kumo filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough for weekly Ichimoku calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1w Indicators: Weekly Ichimoku for Kumo filter ===
    # Weekly Tenkan-sen (9-period)
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    # Weekly Kijun-sen (26-period)
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    # Weekly Senkou Span A
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Weekly Senkou Span B (52-period)
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (period52_high_1w + period52_low_1w) / 2
    
    # Weekly Kumo is bullish when Senkou Span A > Senkou Span B
    weekly_kumo_bullish = senkou_a_1w > senkou_b_1w
    weekly_kumo_bearish = senkou_a_1w < senkou_b_1w
    
    # Align weekly Kumo filter to 6h timeframe
    weekly_kumo_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_kumo_bullish.astype(float))
    weekly_kumo_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_kumo_bearish.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 52 periods needed)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(weekly_kumo_bullish_aligned[i]) or np.isnan(weekly_kumo_bearish_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        price = close[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        wk_bullish = weekly_kumo_bullish_aligned[i] > 0.5
        wk_bearish = weekly_kumo_bearish_aligned[i] > 0.5
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Kumo or Tenkan < Kijun (momentum weakening)
            if price < kumo_bottom_val or tenkan_val < kijun_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Kumo or Tenkan > Kijun (momentum weakening)
            if price > kumo_top_val or tenkan_val > kijun_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Kumo (above cloud) AND Tenkan > Kijun (bullish momentum) AND weekly Kumo bullish
            if price > kumo_top_val and tenkan_val > kijun_val and wk_bullish:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Kumo (below cloud) AND Tenkan < Kijun (bearish momentum) AND weekly Kumo bearish
            elif price < kumo_bottom_val and tenkan_val < kijun_val and wk_bearish:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTKCross_WeeklyKumoFilter_V1"
timeframe = "6h"
leverage = 1.0