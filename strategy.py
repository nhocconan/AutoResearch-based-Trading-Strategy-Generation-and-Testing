#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian(20) breakout with 1-day Ichimoku cloud filter and volume confirmation.
Long when price breaks above Donchian upper band and price > Kumo (cloud top) from 1d.
Short when price breaks below Donchian lower band and price < Kumo (cloud bottom) from 1d.
Exit when price crosses opposite Donchian band or Kumo flips.
Uses Ichimoku cloud from daily timeframe as trend filter to avoid counter-trend trades.
Designed for low trade frequency by requiring both breakout and trend alignment.
Works in bull markets (follows upward breaks) and bear markets (follows downward breaks).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data for Ichimoku cloud - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Kumo (cloud) top and bottom
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku cloud to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top.values)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band AND price > Kumo top (uptrend filter)
            if (close[i] > highest_high[i] and close[i] > kumo_top_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band AND price < Kumo bottom (downtrend filter)
            elif (close[i] < lowest_low[i] and close[i] < kumo_bottom_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian band or Kumo flips (trend change)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian lower band OR Kumo flips (price < Kumo bottom)
                if close[i] < lowest_low[i] or close[i] < kumo_bottom_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian upper band OR Kumo flips (price > Kumo top)
                if close[i] > highest_high[i] or close[i] > kumo_top_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_IchimokuCloud_Volume"
timeframe = "6h"
leverage = 1.0