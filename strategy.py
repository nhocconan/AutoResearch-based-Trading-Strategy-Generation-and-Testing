#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeRegime
Hypothesis: Ichimoku TK cross on 6h with 1d cloud filter (price above/below 1d cloud for trend bias) and volume confirmation (ATR ratio > 1.0). Uses discrete sizing 0.25 to limit trades (~15-25/year). Works in bull/bear via 1d trend and cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align 1d Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h TK cross (Tenkan/Kijun crossover)
    tk_cross = tenkan_aligned - kijun_aligned
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0
    tk_cross_up = (tk_cross > 0) & (tk_cross_prev <= 0)
    tk_cross_down = (tk_cross < 0) & (tk_cross_prev >= 0)
    
    # Calculate cloud (Senkou Span A/B) - cloud top is max, bottom is min
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio, 26 for Kijun, 52 for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = atr_ratio[i] > 1.0
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        size = fixed_size
        
        # Determine price position relative to cloud
        above_cloud = close_val > cloud_top_val
        below_cloud = close_val < cloud_bottom_val
        in_cloud = (close_val >= cloud_bottom_val) & (close_val <= cloud_top_val)
        
        if position == 0:
            # Flat - look for entry
            # Long: TK cross up + price above cloud + volume spike
            if tk_up and above_cloud and vol_spike:
                signals[i] = size
                position = 1
            # Short: TK cross down + price below cloud + volume spike
            elif tk_down and below_cloud and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            # Exit: price re-enters cloud or TK cross down
            if in_cloud or tk_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # Exit: price re-enters cloud or TK cross up
            if in_cloud or tk_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0