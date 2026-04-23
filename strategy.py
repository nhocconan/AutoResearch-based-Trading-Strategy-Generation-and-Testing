#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + 1d Weekly Pivot + Volume Confirmation.
Long when price > Kumo cloud AND Tenkan > Kijun AND price breaks above Weekly R1 AND volume > 1.5x 20-period average.
Short when price < Kumo cloud AND Tenkan < Kijun AND price breaks below Weekly S1 AND volume > 1.5x 20-period average.
Exit when price re-enters Kumo cloud or Tenkan/Kijun cross reverses.
Ichimoku provides trend, support/resistance, and momentum in one system. Weekly pivots add higher-timeframe structure.
Volume confirmation filters weak signals. Designed for 6h timeframe to capture medium-term swings in both bull and bear markets.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Ichimoku - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumioffset = 26  # Kumo cloud plotted 26 periods ahead
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Kumo cloud boundaries (Senkou Span A and B)
    # Kumo is plotted 26 periods ahead, so we need to shift back for current price comparison
    senkou_span_a_shifted = senkou_span_a.shift(kumioffset)
    senkou_span_b_shifted = senkou_span_b.shift(kumioffset)
    
    # Load 1d data for Weekly Pivots - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Weekly Pivot points from previous 1d bar
    # Standard floor pivot: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    weekly_r1 = 2 * pivot_point - prev_low
    weekly_s1 = 2 * pivot_point - prev_high
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not ready
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(senkou_span_a_shifted.iloc[i]) or np.isnan(senkou_span_b_shifted.iloc[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        senkou_a = senkou_span_a_shifted.iloc[i]
        senkou_b = senkou_span_b_shifted.iloc[i]
        vol_ma_val = vol_ma[i]
        
        # Kumo cloud: price above/below both Senkou Span A and B
        kumo_top = max(senkou_a, senkou_b)
        kumo_bottom = min(senkou_a, senkou_b)
        price_above_kumo = price > kumo_top
        price_below_kumo = price < kumo_bottom
        
        if position == 0:
            # Long: price > Kumo AND Tenkan > Kijun AND price breaks above Weekly R1 AND volume spike
            if (price_above_kumo and 
                tenkan > kijun and 
                price > weekly_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price < Kumo AND Tenkan < Kijun AND price breaks below Weekly S1 AND volume spike
            elif (price_below_kumo and 
                  tenkan < kijun and 
                  price < weekly_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price re-enters Kumo OR Tenkan/Kijun cross reverses (Tenkan < Kijun)
                if not price_above_kumo or tenkan < kijun:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price re-enters Kumo OR Tenkan/Kijun cross reverses (Tenkan > Kijun)
                if not price_below_kumo or tenkan > kijun:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Kumo_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0