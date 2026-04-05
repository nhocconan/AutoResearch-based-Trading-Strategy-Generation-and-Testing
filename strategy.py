#!/usr/bin/env python3
"""
exp_7571_6d_2025_06_07_v1
Hypothesis: 6-hour Williams Alligator with 1-day Ichimoku Cloud filter.
Williams Alligator uses smoothed medians (Jaw, Teeth, Lips) to identify trends.
Long when Lips > Teeth > Jaw (bullish alignment) and price above Ichimoku Cloud (1d).
Short when Lips < Teeth < Jaw (bearish alignment) and price below Ichimoku Cloud (1d).
Requires price to be outside cloud to avoid whipsaws in sideways markets.
Ichimoku Cloud acts as dynamic support/resistance filter from higher timeframe.
Targets 50-150 total trades over 4 years with strict alignment conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7571_6d_2025_06_07_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Ichimoku Cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).max() + 
                  pd.Series(low_1d).rolling(window=ICHIMOKU_TENKAN, min_periods=ICHIMOKU_TENKAN).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).max() + 
                 pd.Series(low_1d).rolling(window=ICHIMOKU_KIJUN, min_periods=ICHIMOKU_KIJUN).min()) / 2
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).max() + 
                     pd.Series(low_1d).rolling(window=ICHIMOKU_SENKOU_B, min_periods=ICHIMOKU_SENKOU_B).min()) / 2
    
    # Ichimoku components as arrays
    tenkan_sen_vals = tenkan_sen.values
    kijun_sen_vals = kijun_sen.values
    senkou_span_a_vals = senkou_span_a.values
    senkou_span_b_vals = senkou_span_b.values
    
    # Align Ichimoku to 6h timeframe (shifted by 1 for completed daily bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_vals)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_vals)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_vals)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_vals)
    
    # Calculate Ichimoku Cloud boundaries
    ichimoku_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    ichimoku_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Calculate Williams Alligator on 6h data
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (Blue Line): 13-period SMMA smoothed 8 periods
    jaw = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_JAW, min_periods=ALLIGATOR_PERIOD_JAW).mean()
    jaw = jaw.rolling(window=ALLIGATOR_PERIOD_JAW, min_periods=ALLIGATOR_PERIOD_JAW).mean().values  # Smoothed
    
    # Teeth (Red Line): 8-period SMMA smoothed 5 periods
    teeth = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_TEETH, min_periods=ALLIGATOR_PERIOD_TEETH).mean()
    teeth = teeth.rolling(window=ALLIGATOR_PERIOD_TEETH, min_periods=ALLIGATOR_PERIOD_TEETH).mean().values  # Smoothed
    
    # Lips (Green Line): 5-period SMMA smoothed 3 periods
    lips = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_LIPS, min_periods=ALLIGATOR_PERIOD_LIPS).mean()
    lips = lips.rolling(window=ALLIGATOR_PERIOD_LIPS, min_periods=ALLIGATOR_PERIOD_LIPS).mean().values  # Smoothed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW*2, ALLIGATOR_PERIOD_TEETH*2, ALLIGATOR_PERIOD_LIPS*2, 
                ICHIMOKU_SENKOU_B) + 1
    
    for i in range(start, n):
        # Skip if Ichimoku data not available
        if np.isnan(ichimoku_top[i]) or np.isnan(ichimoku_bottom[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Skip if Alligator data not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        price = prices['close'].values[i]
        
        # Williams Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Ichimoku Cloud filter
        price_above_cloud = price > ichimoku_top[i]
        price_below_cloud = price < ichimoku_bottom[i]
        
        # Entry conditions
        long_entry = bullish_alignment and price_above_cloud
        short_entry = bearish_alignment and price_below_cloud
        
        # Exit conditions: when alignment breaks or price enters cloud
        exit_long = not bullish_alignment or (price >= ichimoku_bottom[i] and price <= ichimoku_top[i])
        exit_short = not bearish_alignment or (price >= ichimoku_bottom[i] and price <= ichimoku_top[i])
        
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
        elif position == 1:  # long position
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:  # short position
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals