#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: Use Ichimoku TK (Tenkan/Kijun) cross as entry signal with daily cloud filter (Senkou Span A/B) to determine trend direction. In uptrend (price above daily cloud), take long on TK cross up; in downtrend (price below daily cloud), take short on TK cross down. This captures momentum shifts aligned with higher timeframe trend, reducing whipsaws. Ichimoku is trend-following but the cloud filter adds robustness in sideways markets. Designed for 6h timeframe with 1d cloud filter to limit trades and improve win rate in both bull and bear markets.
"""

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # TK Cross signals: 1 for TK cross up, -1 for TK cross down
    tk_cross = np.zeros(n)
    tk_cross[tenkan_sen > kijun_sen] = 1
    tk_cross[tenkan_sen < kijun_sen] = -1
    # Cross detection: change in signal
    tk_cross_change = np.diff(tk_cross, prepend=tk_cross[0])
    tk_cross_up = (tk_cross_change == 2)  # -1 to 1
    tk_cross_down = (tk_cross_change == -2)  # 1 to -1
    
    # Daily data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate daily Ichimoku components for cloud
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Daily Tenkan and Kijun
    d_tenkan = (pd.Series(d_high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                pd.Series(d_low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    d_kijun = (pd.Series(d_high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
               pd.Series(d_low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Daily Senkou Span A and B
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    d_senkou_b = (pd.Series(d_high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                  pd.Series(d_low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # The cloud is between Senkou Span A and B
    # For trend filter: price above cloud = bullish, below cloud = bearish
    # We need the current cloud values (which are plotted 26 periods ahead)
    # So we use the values calculated 26 periods ago
    d_senkou_a_shifted = np.roll(d_senkou_a, 26)
    d_senkou_b_shifted = np.roll(d_senkou_b, 26)
    # Fill first 26 values with NaN to avoid look-ahead
    d_senkou_a_shifted[:26] = np.nan
    d_senkou_b_shifted[:26] = np.nan
    
    # Cloud top and bottom
    d_cloud_top = np.maximum(d_senkou_a_shifted, d_senkou_b_shifted)
    d_cloud_bottom = np.minimum(d_senkou_a_shifted, d_senkou_b_shifted)
    
    # Align daily cloud to 6h timeframe
    d_cloud_top_aligned = align_htf_to_ltf(prices, df_1d, d_cloud_top)
    d_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, d_cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Ichimoku)
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26
    
    for i in range(start_idx, n):
        # Skip if cloud data not available
        if np.isnan(d_cloud_top_aligned[i]) or np.isnan(d_cloud_bottom_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below daily cloud
        price_above_cloud = close[i] > d_cloud_top_aligned[i]
        price_below_cloud = close[i] < d_cloud_bottom_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above daily cloud (bullish trend)
            if tk_cross_up[i] and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below daily cloud (bearish trend)
            elif tk_cross_down[i] and price_below_cloud:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TK cross in opposite direction OR price crosses cloud middle
            # Cloud middle line
            cloud_middle = (d_cloud_top_aligned[i] + d_cloud_bottom_aligned[i]) / 2
            price_vs_middle = close[i] - cloud_middle
            
            if position == 1:
                # Exit long: TK cross down OR price crosses below cloud middle
                if tk_cross_down[i] or price_vs_middle < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TK cross up OR price crosses above cloud middle
                if tk_cross_up[i] or price_vs_middle > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals