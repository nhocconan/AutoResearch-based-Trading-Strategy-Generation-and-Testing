#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation
    # Long when price above cloud, Tenkan > Kijun, and 1w trend bullish (price > 1w Kumo top)
    # Short when price below cloud, Tenkan < Kijun, and 1w trend bearish (price < 1w Kumo bottom)
    # Exit when price crosses Tenkan-Kijun midpoint or cloud is violated
    # Volume confirmation: current volume > 1.3x 20-bar average
    # Works in bull (trend-aligned entries) and bear (only counter-trend bounces from cloud edges)
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Chikou Span (Lagging Span): not used for signals (requires future data)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b.values)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Ichimoku Cloud for trend filter
    tenkan_1w = (pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                 pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_1w = (pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    senkou_span_a_1w = (tenkan_1w + kijun_1w) / 2
    senkou_span_b_1w = (pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # 1w Kumo top and bottom
    kumo_top_1w = np.maximum(senkou_span_a_1w, senkou_span_b_1w)
    kumo_bottom_1w = np.minimum(senkou_span_a_1w, senkou_span_b_1w)
    
    # Align 1w Kumo levels to 6h timeframe
    kumo_top_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_top_1w.values)
    kumo_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w.values)
    
    # Volume confirmation: current volume > 1.3x 20-bar average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Start after all indicators are valid
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, displacement) + displacement
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kumo_top_1w_aligned[i]) or np.isnan(kumo_bottom_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries (current)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Kumo twist (cloud color): green when Senkou A > Senkou B
        kumotwist_bullish = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = (close[i] >= cloud_bottom) & (close[i] <= cloud_top)
        
        # Tenkan-Kijun relationship
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # 1w trend filter
        trend_bullish = close[i] > kumo_top_1w_aligned[i]
        trend_bearish = close[i] < kumo_bottom_1w_aligned[i]
        
        # Entry conditions
        # Long: price above cloud, Tenkan > Kijun, bullish 1w trend, bullish Kumo twist, volume confirmation
        long_entry = (price_above_cloud and tenkan_above_kijun and trend_bullish and 
                      kumotwist_bullish and volume_confirmed[i] and position != 1)
        
        # Short: price below cloud, Tenkan < Kijun, bearish 1w trend, bearish Kumo twist, volume confirmation
        short_entry = (price_below_cloud and tenkan_below_kijun and trend_bearish and 
                       not kumotwist_bullish and volume_confirmed[i] and position != -1)
        
        # Exit conditions
        # Exit long: price crosses below Tenkan-Kijun midpoint OR price falls below cloud bottom
        tk_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2
        exit_long = (position == 1 and (close[i] < tk_mid or close[i] < cloud_bottom))
        
        # Exit short: price crosses above Tenkan-Kijun midpoint OR price rises above cloud top
        exit_short = (position == -1 and (close[i] > tk_mid or close[i] > cloud_top))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_ichimoku_trend_filter_volume_v1"
timeframe = "6h"
leverage = 1.0