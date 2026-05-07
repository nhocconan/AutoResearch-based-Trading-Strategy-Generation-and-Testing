#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    # Tenkan and Kijun are current values, no shift needed for alignment
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    # Senkou Span A and B are plotted 26 periods ahead, so we need to shift back for current alignment
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=26)
    # Chikou Span is plotted 26 periods behind, so we need to shift forward for current alignment
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span.values, additional_delay_bars=-26)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 24)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, Chikou above price 26 periods ago, volume spike
            tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top[i]
            chikou_above = chikou_span_aligned[i] > close[i]  # Chikou (close 26 periods ago) above current price
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            
            if tk_cross and price_above_cloud and chikou_above and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, Chikou below price 26 periods ago, volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  close[i] < cloud_bottom[i] and chikou_span_aligned[i] < close[i] and vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun or price drops below cloud
            tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            price_below_cloud = close[i] < cloud_top[i]
            
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun or price rises above cloud
            tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_bottom[i]
            
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku Cloud breakout with daily trend and volume confirmation
# - Ichimoku provides comprehensive trend, support/resistance, and momentum signals
# - TK cross signals momentum shift, cloud acts as dynamic support/resistance
# - Chikou filter ensures trend confirmation (avoids false signals)
# - Volume spike (2x average) confirms institutional participation
# - Works in bull markets (TK cross up + price above cloud) and bear markets (TK cross down + price below cloud)
# - Exit on TK cross reversal or price re-entering cloud
# - Position size 0.25 targets 15-35 trades/year, avoiding fee drag
# - Uses daily Ichimoku for higher timeframe structure, avoiding whipsaws in lower timeframes