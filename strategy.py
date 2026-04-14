# 12h_1wPivot_1dIchimoku_CloudFilter_v1
# Hypothesis: Use weekly Ichimoku cloud as primary trend filter on 12h timeframe.
# Enter long when price is above cloud and breaks above 1-day Ichimoku conversion line with volume confirmation.
# Enter short when price is below cloud and breaks below 1-day Ichimoku base line with volume confirmation.
# Exit when price returns to the opposite Ichimoku line or trend changes.
# Weekly Ichimoku provides strong trend filter to avoid counter-trend trades in both bull and bear markets.
# Daily Ichimoku provides timely entry/exit signals.
# Volume confirmation reduces false breakouts.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Ichimoku cloud (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = np.concatenate([np.full(26, np.nan), close_1w[:-26]])
    
    # Align weekly Ichimoku components to 12h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1w, chikou_span)
    
    # Load daily data ONCE for Ichimoku entry/exit signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Align daily Ichimoku components to 12h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(52, 26, 20)  # Need weekly Ichimoku and daily Ichimoku and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or
            np.isnan(tenkan_sen_1d_aligned[i]) or
            np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below weekly cloud
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for entries
            # Long: price above weekly cloud AND breaks above daily Tenkan-sen with volume
            if (price_above_cloud and 
                close[i] > tenkan_sen_1d_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below weekly cloud AND breaks below daily Kijun-sen with volume
            elif (price_below_cloud and 
                  close[i] < kijun_sen_1d_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below daily Kijun-sen or returns to weekly cloud
            if (close[i] < kijun_sen_1d_aligned[i] or 
                close[i] < cloud_top):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above daily Tenkan-sen or returns to weekly cloud
            if (close[i] > tenkan_sen_1d_aligned[i] or 
                close[i] > cloud_bottom):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wPivot_1dIchimoku_CloudFilter_v1"
timeframe = "12h"
leverage = 1.0