#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_weekly_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we use current price vs cloud from 26 periods ago
    # We'll use the cloud (Senkou Span) from 26 periods ago for current price comparison
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku calculation period
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR Tenkan-Kijun cross down
            if close[i] < lower_cloud or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR Tenkan-Kijun cross up
            if close[i] > upper_cloud or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above cloud AND Tenkan crosses above Kijun
            if (close[i] > upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below cloud AND Tenkan crosses below Kijun
            elif (close[i] < lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals