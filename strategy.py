#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
# Hypothesis: Ichimoku Tenkan/Kijun cross on 6h with 1d trend filter (price > Kumo) and volume confirmation.
# Uses TK cross for entry, Kumo cloud for trend filter, and volume surge for confirmation.
# Designed for low trade frequency (15-30/year) to minimize fee drag while capturing trend continuations.
# Works in both bull and bear markets by using Kumo as dynamic support/resistance.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Kumo cloud calculation (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Ichimoku components on 6d (converted from daily) ---
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    # Kijun-sen (Base Line): (26-period high + low)/2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    
    # Calculate Tenkan and Kijun on 6d data
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A and B
    senkou_span_a = ((tenkan + kijun) / 2)
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Shift Senkou spans forward by 26 periods (for cloud)
    senkou_span_a = np.roll(senkou_span_a, 26)
    senkou_span_b = np.roll(senkou_span_b, 26)
    senkou_span_a[:26] = np.nan
    senkou_span_b[:26] = np.nan
    
    # Align Ichimoku components to 6h (already on 6h timeframe)
    tenkan_aligned = tenkan  # Already aligned
    kijun_aligned = kijun
    senkou_span_a_aligned = senkou_span_a
    senkou_span_b_aligned = senkou_span_b
    
    # --- 1d trend filter: price vs Kumo (cloud) ---
    # Get daily OHLC for Kumo calculation
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Kumo on daily
    d_period9_high = pd.Series(d_high).rolling(window=9, min_periods=9).max().values
    d_period9_low = pd.Series(d_low).rolling(window=9, min_periods=9).min().values
    d_tenkan = (d_period9_high + d_period9_low) / 2
    
    d_period26_high = pd.Series(d_high).rolling(window=26, min_periods=26).max().values
    d_period26_low = pd.Series(d_low).rolling(window=26, min_periods=26).min().values
    d_kijun = (d_period26_high + d_period26_low) / 2
    
    d_senkou_span_a = ((d_tenkan + d_kijun) / 2)
    d_period52_high = pd.Series(d_high).rolling(window=52, min_periods=52).max().values
    d_period52_low = pd.Series(d_low).rolling(window=52, min_periods=52).min().values
    d_senkou_span_b = (d_period52_high + d_period52_low) / 2
    
    # Kumo top and bottom (Senkou Span A/B)
    kumo_top = np.maximum(d_senkou_span_a, d_senkou_span_b)
    kumo_bottom = np.minimum(d_senkou_span_a, d_senkou_span_b)
    
    # Align Kumo to 6h
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    # --- ATR for volatility and trailing stop ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Volume confirmation (2x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kumo_top_aligned[i]) or
            np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Ichimoku signals
        tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price vs Kumo (cloud)
        price_above_kumo = close[i] > kumo_top_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: TK bullish cross, price above Kumo, volume surge
            if tk_cross_bullish and price_above_kumo and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: TK bearish cross, price below Kumo, volume surge
            elif tk_cross_bearish and price_below_kumo and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Exit conditions: TK bearish cross OR price drops below Kumo bottom
                if tk_cross_bearish or close[i] < kumo_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Exit conditions: TK bullish cross OR price rises above Kumo top
                if tk_cross_bullish or close[i] > kumo_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals