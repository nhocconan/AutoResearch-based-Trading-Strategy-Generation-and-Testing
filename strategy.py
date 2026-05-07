#!/usr/bin/env python3
name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
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
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross above cloud with volume
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            
            if tk_cross_up and price_above_cloud and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below cloud with volume
            elif tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]:
                price_below_cloud = close[i] < cloud_bottom
                if price_below_cloud and vol_condition:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: TK cross down or price enters cloud
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_in_cloud = close[i] < cloud_top and close[i] > cloud_bottom
            
            if tk_cross_down or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price enters cloud
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_in_cloud = close[i] < cloud_top and close[i] > cloud_bottom
            
            if tk_cross_up or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with daily cloud filter and volume confirmation
# - Uses daily Ichimoku cloud as dynamic support/resistance and trend filter
# - Tenkan/Kijun cross above/below cloud with volume confirms momentum
# - Works in both bull (buy TK cross above cloud) and bear (sell TK cross below cloud)
# - Cloud acts as natural trend filter: only trade when price is outside cloud
# - Volume confirmation (1.5x average) ensures institutional participation
# - Exit on TK cross reversal or price re-entering cloud
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Ichimoku is proven effective in trending and ranging markets via cloud filter