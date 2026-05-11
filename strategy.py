#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Uses Ichimoku Tenkan/Kijun cross for entry with daily cloud filter for trend direction.
Long when TK crosses above AND price above daily cloud; Short when TK crosses below AND price below daily cloud.
Uses volume confirmation to avoid false signals. Designed for low trade frequency by requiring multiple confluence factors.
Ichimoku works well in both trending and ranging markets, making it suitable for BTC/ETH across market regimes.
"""

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku and cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Ichimoku Components (9, 26, 52 periods) on 1d ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_b = senkou_b.values
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # --- Volume Spike Detection (24-period average) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Ichimoku (52 periods)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        # Determine cloud boundaries
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross signals
        tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                       tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                         tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        if position == 0:
            # Long: TK cross up + price above cloud + volume spike
            if (tk_cross_up and 
                close[i] > upper_cloud and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + volume spike
            elif (tk_cross_down and 
                  close[i] < lower_cloud and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite TK cross or loss of cloud position
            if position == 1:
                # Exit long: TK cross down OR price drops below cloud
                if (tk_cross_down or close[i] < lower_cloud):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TK cross up OR price rises above cloud
                if (tk_cross_up or close[i] > upper_cloud):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals