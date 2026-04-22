#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d filter and volume confirmation.
# Uses Ichimoku TK Cross (Tenkan/Kijun) as entry signal, filtered by price above/below Kumo (cloud) from 1d timeframe.
# Volume spike confirms momentum. Works in bull/bear via cloud direction filter.
# Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku (cloud) and volume context (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods back (not used for entry)
    
    # Align Ichimoku components to 6h timeframe (waits for 1d bar to close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish TK Cross: Tenkan crosses above Kijun
            tk_cross_bullish = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
            # Bearish TK Cross: Tenkan crosses below Kijun
            tk_cross_bearish = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
            
            # Long: bullish TK Cross + price above cloud + volume spike
            if tk_cross_bullish and (close[i] > cloud_top[i]) and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK Cross + price below cloud + volume spike
            elif tk_cross_bearish and (close[i] < cloud_bottom[i]) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on bearish TK Cross or price drops below cloud
                tk_cross_bearish = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
                if tk_cross_bearish or (close[i] < cloud_bottom[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish TK Cross or price rises above cloud
                tk_cross_bullish = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
                if tk_cross_bullish or (close[i] > cloud_top[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0