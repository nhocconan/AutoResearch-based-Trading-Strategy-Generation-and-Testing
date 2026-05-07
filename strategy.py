#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_1wTrend_1dVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need 52 weeks for Ichimoku
        return np.zeros(n)
    
    # Load daily data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_9 = df_1w['high'].rolling(window=9, min_periods=9).max().values
    low_9 = df_1w['low'].rolling(window=9, min_periods=9).min().values
    high_26 = df_1w['high'].rolling(window=26, min_periods=26).max().values
    low_26 = df_1w['low'].rolling(window=26, min_periods=26).min().values
    high_52 = df_1w['high'].rolling(window=52, min_periods=52).max().values
    low_52 = df_1w['low'].rolling(window=52, min_periods=52).min().values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (high_9 + low_9) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (high_26 + low_26) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Daily volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku cloud: Senkou A and B form the cloud
        # Cloud top is max(Senkou A, Senkou B), bottom is min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross above cloud with volume confirmation
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            if tk_cross_bullish and price_above_cloud and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below cloud with volume confirmation
            elif tk_cross_bullish == False and tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]:
                tk_cross_bearish = True
                price_below_cloud = close[i] < cloud_bottom
                if tk_cross_bearish and price_below_cloud and vol_condition:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: TK cross below cloud or price enters cloud
            tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_in_or_below_cloud = close[i] <= cloud_top
            if tk_cross_bearish or price_in_or_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross above cloud or price enters cloud
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_in_or_above_cloud = close[i] >= cloud_bottom
            if tk_cross_bullish or price_in_or_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Ichimoku TK Cross with cloud filter and volume confirmation
# - Weekly Ichimoku provides multi-dimensional support/resistance (cloud) and momentum (TK cross)
# - TK cross above/below cloud with volume confirms strong momentum in direction of trend
# - Cloud acts as dynamic support/resistance - avoids whipsaws in ranging markets
# - Volume confirmation (1.5x average) filters false signals
# - Works in both bull (buy TK cross above cloud in uptrend) and bear (sell TK cross below cloud in downtrend)
# - Exit when TK reverses or price re-enters cloud
# - Position size 0.25 targets ~15-35 trades/year, well within limits
# - Uses weekly Ichimoku for stability and major trend identification
# - Novel for 6h timeframe - combines Ichimoku's predictive power with volume confirmation
# - Aims for 60-140 total trades over 4 years (15-35/year) to avoid fee drag
# - Ichimoku cloud filters out false breaks during consolidation periods
# - TK cross provides timely entry signals with trend confirmation from weekly timeframe
# - Volume requirement ensures institutional participation in breakouts
# - Designed to capture major trend changes while avoiding false signals in choppy markets