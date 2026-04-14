#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud with 6h Tenkan/Kijun cross and volume confirmation.
# The Ichimoku Cloud from daily data provides strong support/resistance levels.
# Tenkan/Kijun cross on 6h timeframe provides entry signals in the direction of the daily trend.
# Volume confirmation (>1.3x 20-period average) reduces false signals.
# Works in both bull and bear markets by using daily cloud as trend filter (price above/below cloud).
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Ichimoku Cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku Cloud on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Chikou Span (Lagging Span): not used for signals
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Load 6h data for Tenkan/Kijun cross calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 26:
        return np.zeros(n)
    
    # Calculate Tenkan-sen and Kijun-sen on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (9-period) on 6h
    tenkan_sen_6h = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen_6h = tenkan_sen_6h.values
    
    # Kijun-sen (26-period) on 6h
    kijun_sen_6h = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen_6h = kijun_sen_6h.values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(52, 26, 20)  # Need Ichimoku (52), 6h Kijun (26), volume MA (20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # 6h Tenkan/Kijun cross
        tk_cross_bull = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bear = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross AND volume confirmation
            if (close[i] > upper_cloud and 
                tk_cross_bull and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below cloud AND bearish TK cross AND volume confirmation
            elif (close[i] < lower_cloud and 
                  tk_cross_bear and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below cloud OR bearish TK cross
            if (close[i] < lower_cloud or 
                tk_cross_bear):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above cloud OR bullish TK cross
            if (close[i] > upper_cloud or 
                tk_cross_bull):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dIchimoku_6hTKCross_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0