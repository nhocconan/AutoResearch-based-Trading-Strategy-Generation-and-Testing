#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku Cloud with 1w trend filter and volume confirmation.
    # Long when price > 1w Ichimoku Cloud (bullish trend) AND TK Cross bullish AND 6h volume > 1.5x 20-period MA.
    # Short when price < 1w Ichimoku Cloud (bearish trend) AND TK Cross bearish AND 6h volume > 1.5x 20-period MA.
    # Exit when price re-enters the cloud (mean reversion).
    # Uses Ichimoku for trend structure, TK Cross for momentum, volume for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku Cloud calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku parameters: tenkan=9, kijun=26, senkou_span_b=52, displacement=26
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    # Get 6h data for volume confirmation and price
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Align 6h close for cloud comparison
    close_6h_aligned = align_htf_to_ltf(prices, df_6h, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(close_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_spike = volume_6h_aligned[i] > 1.5 * vol_ma_6h_aligned[i]
        
        # Ichimoku Cloud boundaries (using Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = close_6h_aligned[i] > upper_cloud
        price_below_cloud = close_6h_aligned[i] < lower_cloud
        price_in_cloud = (close_6h_aligned[i] >= lower_cloud) & (close_6h_aligned[i] <= upper_cloud)
        
        # TK Cross (Tenkan-sen/Kijun-sen cross)
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        tk_cross_previous_bullish = tenkan_sen_aligned[i-1] > kijun_sen_aligned[i-1]
        tk_cross_previous_bearish = tenkan_sen_aligned[i-1] < kijun_sen_aligned[i-1]
        
        # Bullish TK Cross: Tenkan crosses above Kijun
        bullish_tk_cross = tk_cross_bullish and not tk_cross_previous_bullish
        # Bearish TK Cross: Tenkan crosses below Kijun
        bearish_tk_cross = tk_cross_bearish and not tk_cross_previous_bearish
        
        # Entry conditions
        if price_above_cloud and bullish_tk_cross and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_cloud and bearish_tk_cross and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price re-enters the cloud
        elif price_in_cloud and position != 0:
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

name = "6h_1w_ichimoku_cloud_tk_cross_volume_v1"
timeframe = "6h"
leverage = 1.0