#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with daily trend filter
# Tenkan-sen (9-period) / Kijun-sen (26-period) cross on 6h timeframe
# Cloud (Senkou Span A/B) from daily timeframe acts as trend filter
# Long when Tenkan > Kijun AND price above daily cloud (bullish bias)
# Short when Tenkan < Kijun AND price below daily cloud (bearish bias)
# Exit when Tenkan/Kijun cross reverses
# Ichimoku works in both trending and ranging markets, cloud provides dynamic support/resistance
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 6h and daily data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    
    # Calculate daily Ichimoku cloud components (Senkou Span A/B)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2 (plotted 26 periods ahead)
    tenkan_daily = (pd.Series(high_daily).rolling(window=9, min_periods=9).max() + 
                    pd.Series(low_daily).rolling(window=9, min_periods=9).min()) / 2
    kijun_daily = (pd.Series(high_daily).rolling(window=26, min_periods=26).max() + 
                   pd.Series(low_daily).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a = ((tenkan_daily + kijun_daily) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2 (plotted 26 periods ahead)
    senkou_span_b = ((pd.Series(high_daily).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_daily).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max 52 for Senkou Span B)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long setup: Tenkan > Kijun AND price above cloud (bullish)
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                price > upper_cloud):
                position = 1
                signals[i] = position_size
            # Short setup: Tenkan < Kijun AND price below cloud (bearish)
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  price < lower_cloud):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan/Kijun cross reverses (Tenkan < Kijun)
            if tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan/Kijun cross reverses (Tenkan > Kijun)
            if tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_DailyCloud"
timeframe = "6h"
leverage = 1.0