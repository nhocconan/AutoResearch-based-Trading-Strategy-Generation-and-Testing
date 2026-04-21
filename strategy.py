#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross filtered by 1d cloud color (trend) and volume spike (>1.5x average).
Tenkan-sen (9-period) crossing above/below Kijun-sen (26-period) generates signals.
Only take longs when price is above 1d cloud (bullish trend) and shorts when below (bearish trend).
Volume confirmation reduces false signals. Designed for 50-150 total trades over 4 years.
Works in bull/bear via 1d trend filter and volume spike to avoid chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend/cloud filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Kijun
        return np.zeros(n)
    
    # === Ichimoku Components on 6h chart ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Chikou Span (Lagging Span): not used for signals
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # Align Ichimoku components to 6h timeframe (use previous completed values)
    tenkan_aligned = align_htf_to_ltf(prices, prices.index, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, prices.index, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, prices.index, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, prices.index, senkou_span_b.values)
    
    # === 1d Cloud for Trend Filter ===
    # 1d Ichimoku cloud (same calculation but on daily data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen 1d
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen 1d
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A 1d
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B 1d
    senkou_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # The cloud is between Senkou A and Senkou B
    # Cloud color: green (bullish) when Senkou A > Senkou B, red (bearish) when Senkou A < Senkou B
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d  # True when cloud is bullish
    
    # Align 1d cloud data to 6h timeframe
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d.astype(float))
    
    # === Volume Spike Filter ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma  # volume > 1.5x 20-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Kijun period
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(cloud_bullish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        is_volume_spike = vol_spike[i] if not np.isnan(vol_ma[i]) else False
        is_cloud_bullish = bool(cloud_bullish_aligned[i]) if not np.isnan(cloud_bullish_aligned[i]) else False
        
        if position == 0:
            # Tenkan-Kijun cross signals
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tenkan_curr = tenkan_aligned[i]
            kijun_curr = kijun_aligned[i]
            
            # Bullish cross: Tenkan crosses above Kijun
            bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
            # Bearish cross: Tenkan crosses below Kijun
            bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
            
            # Entry conditions
            if bullish_cross and is_cloud_bullish and is_volume_spike:
                # Price above cloud (additional bullish confirmation)
                if price > cloud_top_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif bearish_cross and not is_cloud_bullish and is_volume_spike:
                # Price below cloud (additional bearish confirmation)
                if price < cloud_bottom_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions for long
            # Tenkan-Kijun death cross or price below cloud
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tenkan_curr = tenkan_aligned[i]
            kijun_curr = kijun_aligned[i]
            bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
            
            if bearish_cross or price < cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            # Tenkan-Kijun golden cross or price above cloud
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tenkan_curr = tenkan_aligned[i]
            kijun_curr = kijun_aligned[i]
            bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
            
            if bullish_cross or price > cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0