# Hypothesis: 6h momentum strategy using 1d Ichimoku Cloud and 12h Volume Confirmation
# Long: Price above Kumo + Tenkan > Kijun + Volume > 1.2x average (12h)
# Short: Price below Kumo + Tenkan < Kijun + Volume > 1.2x average (12h)
# Exit: Opposite Tenkan/Kijun cross or price crosses Kumo
# Uses Ichimoku for trend/momentum with volume filter to avoid false breakouts.
# Designed for 6h timeframe with 12h volume and 1d Ichimoku for multi-timeframe alignment.
# Target: 50-120 total trades over 4 years (12-30/year) with disciplined entries.

#!/usr/bin/env python3
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
    
    # Load 1d data for Ichimoku calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For Ichimoku, the cloud is plotted 26 periods ahead
    senkou_span_a_leading = senkou_span_a.shift(26)
    senkou_span_b_leading = senkou_span_b.shift(26)
    
    # Calculate 12h volume average (20-period)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_leading_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_leading.values)
    senkou_span_b_leading_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_leading.values)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_leading_aligned[i]) or np.isnan(senkou_span_b_leading_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_leading_aligned[i], senkou_span_b_leading_aligned[i])
        cloud_bottom = min(senkou_span_a_leading_aligned[i], senkou_span_b_leading_aligned[i])
        
        # Long entry: Price above cloud + Tenkan > Kijun + Volume confirmation
        if (close[i] > cloud_top and
            tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
            volume[i] > 1.2 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Price below cloud + Tenkan < Kijun + Volume confirmation
        elif (close[i] < cloud_bottom and
              tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
              volume[i] > 1.2 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Tenkan/Kijun cross or price crosses cloud
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_12hVolume_Filter"
timeframe = "6h"
leverage = 1.0