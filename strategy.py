#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud Breakout with Weekly Trend Filter
# Hypothesis: Ichimoku TK crossover above/below cloud on 6h, filtered by weekly Kumo twist,
# captures breakouts in both bull and bear markets. Weekly trend ensures alignment with
# higher timeframe momentum, reducing false signals. Target: 15-30 trades/year (60-120 total).

name = "6h_ichimoku_kumo_breakout_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Kumo twist filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_high_tenkan = pd.Series(high_weekly).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_tenkan = pd.Series(low_weekly).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_high_kijun = pd.Series(high_weekly).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_kijun = pd.Series(low_weekly).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    highest_high_senkou_b = pd.Series(high_weekly).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_senkou_b = pd.Series(low_weekly).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # Kumo twist: bullish when Senkou A > Senkou B, bearish when Senkou A < Senkou B
    kumo_twist_bullish = senkou_span_a > senkou_span_b
    kumo_twist_bearish = senkou_span_a < senkou_span_b
    
    # Align weekly Kumo twist to 6h
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_weekly, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_weekly, kumo_twist_bearish.astype(float))
    
    # Calculate Ichimoku on 6h for entry signals
    period_tenkan_6h = 9
    period_kijun_6h = 26
    period_senkou_b_6h = 52
    
    highest_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    lowest_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_sen_6h = (highest_high_tenkan_6h + lowest_low_tenkan_6h) / 2
    
    highest_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    lowest_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_sen_6h = (highest_high_kijun_6h + lowest_low_kijun_6h) / 2
    
    # Senkou Span A and B for 6h (shifted forward by 26 periods)
    senkou_span_a_6h = (tenkan_sen_6h + kijun_sen_6h) / 2
    highest_high_senkou_b_6h = pd.Series(high).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).max().values
    lowest_low_senkou_b_6h = pd.Series(low).rolling(window=period_senkou_b_6h, min_periods=period_senkou_b_6h).min().values
    senkou_span_b_6h = (highest_high_senkou_b_6h + lowest_low_senkou_b_6h) / 2
    
    # Kumo cloud boundaries (Senkou Span A and B)
    # For plotting, Senkou spans are shifted 26 periods ahead, but for current cloud we use current values
    # The cloud is between Senkou A and Senkou B
    kumotop_6h = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    kumobottom_6h = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need enough data for Ichimoku
        # Skip if required data not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(kumotop_6h[i]) or np.isnan(kumobottom_6h[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TK cross down OR price falls below Kumo
            if tenkan_sen_6h[i] < kijun_sen_6h[i] or close[i] < kumobottom_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: TK cross up OR price rises above Kumo
            if tenkan_sen_6h[i] > kijun_sen_6h[i] or close[i] > kumotop_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: TK cross up AND price above Kumo AND weekly Kumo bullish twist
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                close[i] > kumotop_6h[i] and 
                kumo_twist_bullish_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short: TK cross down AND price below Kumo AND weekly Kumo bearish twist
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  close[i] < kumobottom_6h[i] and 
                  kumo_twist_bearish_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals