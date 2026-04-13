#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud (from 1d) + TK Cross + Volume Confirmation
    # Long when: price > 1d Ichimoku Cloud AND TK Cross bullish AND volume > 1.5x 20-bar avg
    # Short when: price < 1d Ichimoku Cloud AND TK Cross bearish AND volume > 1.5x 20-bar avg
    # Exit when: price crosses 1d Kumo (cloud) midpoint OR TK Cross reverses
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Ichimoku provides dynamic support/resistance; TK Cross gives momentum signal; volume confirms validity.
    # Works in bull (price above cloud with bullish TK) and bear (price below cloud with bearish TK).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (1d)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate Kumo (Cloud) boundaries and midpoint
    # Senkou Span A and B are plotted 26 periods ahead, so we need to shift them back for current price comparison
    # For current price, we use the values that were plotted 26 periods ago
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_mid = (kumo_top + kumo_bottom) / 2
    
    # Calculate TK Cross (Tenkan-sen / Kijun-sen cross)
    tk_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Start loop after sufficient warmup for Ichimoku (52 periods for Senkou Span B)
    start_idx = 52
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(kumo_mid[i]) or
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > kumo_top[i] and tk_bullish[i] and volume_confirmed[i] and position != 1)
        short_entry = (close[i] < kumo_bottom[i] and tk_bearish[i] and volume_confirmed[i] and position != -1)
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < kumo_mid[i] or not tk_bullish[i]))
        exit_short = (position == -1 and (close[i] > kumo_mid[i] or not tk_bearish[i]))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_tk_volume_v1"
timeframe = "6h"
leverage = 1.0