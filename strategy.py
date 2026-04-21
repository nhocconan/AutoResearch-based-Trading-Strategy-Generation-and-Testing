#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation. 
Ichimoku provides dynamic support/resistance and trend direction. The 1d cloud acts as a higher-timeframe trend filter 
to avoid counter-trend trades. Volume spike (>1.3x 20-period MA) confirms momentum. 
This strategy works in both bull and bear markets by only taking trades aligned with the 1d Ichimoku cloud.
Target: 12-35 trades/year (50-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2.0)
    
    # The actual cloud (for current price) is Senkou Span A/B shifted back 26 periods
    # So we need to shift the calculated Senkou lines BACK by 26 to align with current price
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom (Senkou A and B)
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Ichimoku warmup
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) 
            or np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i])
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.3 * vol_ma[i]  # volume spike confirmation
        
        # Bullish TK cross: Tenkan > Kijun
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        # Bearish TK cross: Tenkan < Kijun
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = price > cloud_top_aligned[i]
        price_below_cloud = price < cloud_bottom_aligned[i]
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + volume spike
            if tk_bullish and price_above_cloud and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + volume spike
            elif tk_bearish and price_below_cloud and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross or price drops below cloud
            if tk_bearish or price < cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross or price rises above cloud
            if tk_bullish or price > cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0