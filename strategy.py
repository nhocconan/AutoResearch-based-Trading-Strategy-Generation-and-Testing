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
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (high + low) / 2 over tenkan_period
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen_1d = tenkan_sen_1d.values
    
    # Calculate Kijun-sen (Base Line): (high + low) / 2 over kijun_period
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen_1d = kijun_sen_1d.values
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (high + low) / 2 over senkou_span_b_period
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b_1d = senkou_span_b_1d.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Kumo (Cloud) top and bottom
    kumo_top_6h = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    kumo_bottom_6h = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # Volume filter: current volume > 1.5 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(kumo_top_6h[i]) or 
            np.isnan(kumo_bottom_6h[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma30[i])
        
        # Ichimoku signals
        price_above_kumo = close[i] > kumo_top_6h[i]
        price_below_kumo = close[i] < kumo_bottom_6h[i]
        tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        if position == 0:
            # Long: Price above cloud AND TK cross bullish with volume
            if (price_above_kumo and tk_cross_bullish and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND TK cross bearish with volume
            elif (price_below_kumo and tk_cross_bearish and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below cloud OR TK cross turns bearish
            if (close[i] < kumo_top_6h[i]) or (tk_cross_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above cloud OR TK cross turns bullish
            if (close[i] > kumo_bottom_6h[i]) or (tk_cross_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0