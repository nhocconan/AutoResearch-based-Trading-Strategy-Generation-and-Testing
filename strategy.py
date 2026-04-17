#!/usr/bin/env python3
"""
6h_Ichimoku_CloudBreakout_TrendFilter
Strategy: Ichimoku cloud breakout with weekly trend filter.
- Long when price breaks above Kumo cloud (Tenkan > Kijun) and price > Kumo top + weekly trend up
- Short when price breaks below Kumo cloud (Tenkan < Kijun) and price < Kumo bottom + weekly trend down
- Exit when price returns to Tenkan-Kijun midpoint
- Uses Ichimoku on 6h timeframe with weekly trend filter (Kumo twist from weekly)
- Position size: 0.25
- Designed to capture trends in both bull and bear markets with cloud acting as dynamic support/resistance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used in signals
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo top and bottom (Senkou Span A and B)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Get weekly data for trend filter (Kumo twist)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Ichimoku for trend filter
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, df_1w['close'].values)
    kumo_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumo_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align weekly Ichimoku to 6h timeframe (weekly data needs extra delay for confirmation)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w, additional_delay_bars=1)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w, additional_delay_bars=1)
    kumo_top_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_top_1w, additional_delay_bars=1)
    kumo_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w, additional_delay_bars=1)
    
    # Tenkan-Kijun crossover signals
    tk_cross_up = (tenkan > kijun) & (tenkan <= kijun)  # Crossed up
    tk_cross_down = (tenkan < kijun) & (tenkan >= kijun)  # Crossed down
    
    # Price relative to cloud
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Weekly trend filter: price above/below weekly cloud
    weekly_uptrend = close > kumo_top_1w_aligned
    weekly_downtrend = close < kumo_bottom_1w_aligned
    
    # Exit signal: price returns to Tenkan-Kijun midpoint
    tk_midpoint = (tenkan + kijun) / 2
    return_to_tk = np.abs(close - tk_midpoint) < 0.002 * close  # Within 0.2% of TK midpoint
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 52)  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(kumo_top_1w_aligned[i]) or np.isnan(kumo_bottom_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up[i] and price_above_kumo[i] and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down[i] and price_below_kumo[i] and weekly_downtrend[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to TK midpoint or TK cross down
            if return_to_tk[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to TK midpoint or TK cross up
            if return_to_tk[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0