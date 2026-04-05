#!/usr/bin/env python3
"""
Experiment #7955: 6-hour Ichimoku Cloud with 1-week trend filter.
Hypothesis: Price crossing Tenkan/Kijun lines above/below the Kumo (cloud) on 6h 
with weekly Kumo twist (Senkou Span A > Senkou Span B for bullish, A < B for bearish) 
captures trend continuation with low whipsaw. Weekly trend filter ensures alignment 
with higher timeframe momentum, reducing false signals in ranging markets. 
Ichimoku provides built-in support/resistance and momentum confirmation. 
Target: 75-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7955_6h_ichimoku_1w_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9    # Tenkan-sen (Conversion Line)
KIJUN_PERIOD = 26    # Kijun-sen (Base Line)
SENKOU_PERIOD = 52   # Senkou Span B period
KUMO_SHIFT = 26      # Kumo (cloud) shift forward
TK_CROSS_CONFIRM = 1 # Bars to confirm TK cross
SIGNAL_SIZE = 0.25   # Position size (25% of capital)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = (pd.Series(high_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                     pd.Series(low_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2
    
    # Kumo twist: Senkou Span A > B = bullish twist, A < B = bearish twist
    # Shift forward by KUMO_SHIFT periods (cloud is plotted ahead)
    senkou_span_a_shifted = senkou_span_a.shift(KUMO_SHIFT)
    senkou_span_b_shifted = senkou_span_b.shift(KUMO_SHIFT)
    kumo_bullish_twist = senkou_span_a_shifted > senkou_span_b_shifted  # Bullish twist
    kumo_bearish_twist = senkou_span_a_shifted < senkou_span_b_shifted  # Bearish twist
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    kumo_bullish_aligned = align_htf_to_ltf(prices, df_1w, kumo_bullish_twist.values)
    kumo_bearish_aligned = align_htf_to_ltf(prices, df_1w, kumo_bearish_twist.values)
    
    # Calculate LTF Ichimoku (6h) for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen and Kijun-sen on 6h
    tenkan_sen_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                     pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                    pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # TK Cross signals (bullish when Tenkan crosses above Kijun)
    tk_cross_bullish = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_bearish = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))
    
    # Kumo (cloud) boundaries on 6h - shifted forward
    senkou_span_a_6h = ((tenkan_sen_6h + kijun_sen_6h) / 2).shift(KUMO_SHIFT)
    senkou_span_b_6h = (pd.Series(high).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                        pd.Series(low).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2
    senkou_span_b_6h = senkou_span_b_6h.shift(KUMO_SHIFT)
    
    # Price above/below cloud
    price_above_kumo = (close > senkou_span_a_6h) & (close > senkou_span_b_6h)
    price_below_kumo = (close < senkou_span_a_6h) & (close < senkou_span_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available (NaN values from alignment)
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_bullish_aligned[i]) or np.isnan(kumo_bearish_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Get weekly trend bias from Kumo twist
        bullish_weekly = kumo_bullish_aligned[i] == 1
        bearish_weekly = kumo_bearish_aligned[i] == 1
        
        # Entry conditions require weekly trend alignment
        long_entry = (bullish_weekly and 
                     tk_cross_bullish.iloc[i] and 
                     price_above_kumo.iloc[i])
        
        short_entry = (bearish_weekly and 
                      tk_cross_bearish.iloc[i] and 
                      price_below_kumo.iloc[i])
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when TK cross turns bearish OR price drops below cloud
            if tk_cross_bearish.iloc[i] or not price_above_kumo.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when TK cross turns bullish OR price rises above cloud
            if tk_cross_bullish.iloc[i] or not price_below_kumo.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals