#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with TK Cross and 1d trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price above/below Kumo cloud).
- Entry: Long when Tenkan-sen crosses above Kijun-sen AND price above Kumo (bullish bias);
         Short when Tenkan-sen crosses below Kijun-sen AND price below Kumo (bearish bias).
- Exit: Opposite TK cross OR price crosses Kumo in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Ichimoku provides dynamic support/resistance (Kumo cloud) and momentum (TK cross).
- Works in bull markets (buy TK crosses above cloud) and bear markets (sell TK crosses below cloud).
- Estimated trades: ~100 total over 4 years (~25/year) based on TK cross frequency with cloud filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max().values
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max().values
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max().values
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: price above/below Kumo cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d data for trend filter
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Kumo cloud boundaries (Senkou Span A and B)
    # The cloud is between Senkou A and Senkou B
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Trend filter: price above cloud = bullish, price below cloud = bearish
    close_1d = df_1d['close'].values
    bullish_trend = close_1d > kumo_top_1d
    bearish_trend = close_1d < kumo_bottom_1d
    
    # Align 1d indicators to 6h timeframe
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend, additional_delay_bars=1)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend, additional_delay_bars=1)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d, additional_delay_bars=1)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d, additional_delay_bars=1)
    
    # Calculate Ichimoku on 6h data for entry signals (TK cross)
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # TK cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Handle first element for roll
    tk_cross_above[0] = False
    tk_cross_below[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (max period 52)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite TK cross OR price crosses Kumo in opposite direction
        if position != 0:
            # Exit long: TK cross below OR price falls below Kumo bottom
            if position == 1:
                if tk_cross_below[i] or curr_close < kumo_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: TK cross above OR price rises above Kumo top
            elif position == -1:
                if tk_cross_above[i] or curr_close > kumo_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: TK cross in direction of 1d trend filter
        if position == 0:
            # Long: TK cross above AND bullish 1d trend (price above cloud)
            if tk_cross_above[i] and bullish_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND bearish 1d trend (price below cloud)
            elif tk_cross_below[i] and bearish_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0