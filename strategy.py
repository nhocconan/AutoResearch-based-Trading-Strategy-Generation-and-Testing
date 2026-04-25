#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1dTrendFilter_v2
Hypothesis: Trade 6h Ichimoku TK cross in direction of 1d cloud (price above/below cloud) with volume confirmation.
Uses Ichimoku (Tenkan=9, Kijun=26, Senkou Span B=52) on 6h timeframe. Only long when TK cross bullish AND price > 1d Senkou Span A/B AND volume > 1.5 * ATR6h.
Only short when TK cross bearish AND price < 1d Senkou Span A/B AND volume > 1.5 * ATR6h.
Discrete sizing 0.25 to limit fee drag. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Senkou Span A and B for cloud
    # Senkou Span A = (Tenkan + Kijun) / 2 plotted 26 periods ahead
    # Senkou Span B = (Highest High + Lowest Low)/2 over 52 periods plotted 26 periods ahead
    # For simplicity, we use current values (aligned will handle delay)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    
    # Tenkan-sen (9-period): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(h_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(l_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (26-period): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(h_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(l_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A = (Tenkan + Kijun) / 2
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2).values
    # Senkou Span B = (52-period high + 52-period low)/2
    senkou_span_b_1d = ((pd.Series(h_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(l_1d).rolling(window=52, min_periods=52).min()) / 2).values
    
    # Align 1d cloud to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (9-period)
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (26-period)
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # TK Cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    # Calculate ATR for volume confirmation (using 6h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start index: need warmup for Ichimoku and ATR
    start_idx = max(26, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(tk_bullish[i]) or np.isnan(tk_bearish[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR (moderate filter)
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine 1d cloud: price above/both spans = bullish, below/both = bearish
        # Cloud top = max(Senkou A, Senkou B), cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            bars_since_entry = 0
            # Long setup: TK bullish AND price above cloud AND volume confirm
            long_setup = tk_bullish[i] and price_above_cloud and volume_confirm
            
            # Short setup: TK bearish AND price below cloud AND volume confirm
            short_setup = tk_bearish[i] and price_below_cloud and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK turns bearish OR price drops below cloud OR min hold (4 bars) + adverse move
            if bars_since_entry >= 4:
                if (not tk_bullish[i]) or (not price_above_cloud):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            bars_since_entry += 1
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK turns bullish OR price rises above cloud OR min hold (4 bars) + adverse move
            if bars_since_entry >= 4:
                if (not tk_bearish[i]) or (not price_below_cloud):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrendFilter_v2"
timeframe = "6h"
leverage = 1.0