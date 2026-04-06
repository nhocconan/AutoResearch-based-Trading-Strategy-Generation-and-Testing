#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d filter
# Long when price > Kumo (cloud), Tenkan > Kijun, and 1d close > 1d Kumo top
# Short when price < Kumo, Tenkan < Kijun, and 1d close < 1d Kumo bottom
# Uses Kumo as dynamic support/resistance and 1d trend filter for bias
# Targets 50-150 trades over 4 years (12-37/year) with size 0.25

name = "6h_ichimoku_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
              pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
             pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    
    # Kumo (Cloud) boundaries
    kumotop = np.maximum(senkou_a, senkou_b)
    kumobottom = np.minimum(senkou_a, senkou_b)
    
    # 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku cloud
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    kumotop_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumobottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 6h Ichimoku to 6s timeframe (no shift needed as values are for current bar)
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan.values) if False else tenkan
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun.values) if False else kijun
    kumotop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kumotop) if False else kumotop
    kumobottom_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kumobottom) if False else kumobottom
    
    # Actually compute Ichimoku properly on 6h data (above already done)
    tenkan_6h = tenkan
    kijun_6h = kijun
    kumotop_6h = kumotop
    kumobottom_6h = kumobottom
    
    # Align 1d Ichimoku to 6h timeframe
    kumotop_1d_aligned = align_htf_to_ltf(prices, df_1d, kumotop_1d)
    kumobottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumobottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumotop_6h[i]) or np.isnan(kumobottom_6h[i]) or
            np.isnan(kumotop_1d_aligned[i]) or np.isnan(kumobottom_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Kumo or Tenkan/Kijun cross reverses
        if position == 1:  # long position
            if (close[i] <= kumotop_6h[i] or  # price below cloud top
                tenkan_6h[i] < kijun_6h[i]):   # Tenkan below Kijun
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] >= kumobottom_6h[i] or  # price above cloud bottom
                tenkan_6h[i] > kijun_6h[i]):    # Tenkan above Kijun
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with 1d trend filter
            # Bullish: price above cloud, Tenkan > Kijun, and 1d close > 1d cloud top
            if (close[i] > kumotop_6h[i] and 
                tenkan_6h[i] > kijun_6h[i] and
                close_1d[i] > kumotop_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: price below cloud, Tenkan < Kijun, and 1d close < 1d cloud bottom
            elif (close[i] < kumobottom_6h[i] and 
                  tenkan_6h[i] < kijun_6h[i] and
                  close_1d[i] < kumobottom_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals