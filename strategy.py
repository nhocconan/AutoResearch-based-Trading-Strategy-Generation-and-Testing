#!/usr/bin/env python3
"""
4h_Ichimoku_Kumo_Breakout_12hTrend_Volume
Hypothesis: On 4-hour timeframe, enter long when price breaks above Kumo cloud with volume surge and 12h uptrend (price above Kumo top), short when price breaks below Kumo cloud with volume surge and 12h downtrend. Exit on opposite Kumo break. Uses 12h Kumo trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) to minimize fee decay in both bull and bear markets. Ichimoku Kumo provides dynamic support/resistance that adapts to volatility, working well in trending and ranging conditions.
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
    
    # Get 12h data for Kumo cloud (trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_12h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_12h).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_12h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_12h).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(2)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = ((pd.Series(high_12h).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_12h).rolling(window=52, min_periods=52).min()) / 2).shift(2)
    
    # Align Ichimoku components to 4h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b.values)
    
    # Kumo cloud boundaries
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 12h trend: bullish when price > Kumo top, bearish when price < Kumo bottom
    # Note: Using close price for trend determination
    twelve_h_uptrend = close > kumo_top
    twelve_h_downtrend = close < kumo_bottom
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 12h Kumo trend alignment and volume surge
        long_entry = close[i] > kumo_top[i] and twelve_h_uptrend[i] and volume_surge[i]
        short_entry = close[i] < kumo_bottom[i] and twelve_h_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Kumo break with volume surge
        long_exit = close[i] < kumo_bottom[i] and volume_surge[i]
        short_exit = close[i] > kumo_top[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Ichimoku_Kumo_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0