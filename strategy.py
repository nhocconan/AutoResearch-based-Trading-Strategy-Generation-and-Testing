#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_v1
Hypothesis: 6h Ichimoku TK cross with 1d trend filter (price above/below cloud) and volume confirmation.
- Long when TK crosses up, price above 1d cloud, and volume spike
- Short when TK crosses down, price below 1d cloud, and volume spike
- Uses Ichimoku components: Tenkan (9), Kijun (26), Senkou A/B (52 displacement)
- 1d cloud acts as major support/resistance filter
- Volume spike (2.0x 20-period average) confirms institutional participation
- Designed for low frequency (target 12-25 trades/year) with proven edge in ranging/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d trend: price above/both cloud lines
    # In Ichimoku, cloud is between Senkou A and Senkou B
    # Trend = 1 when price > max(Senkou A, Senkou B) (above cloud)
    # Trend = -1 when price < min(Senkou A, Senkou B) (below cloud)
    # Trend = 0 when price inside cloud
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    trend_1d = np.where(close > cloud_top, 1,
                       np.where(close < cloud_bottom, -1, 0))
    
    # Calculate TK cross (Tenkan/Kijun crossover)
    # TK cross up: previous Tenkan < previous Kijun AND current Tenkan >= current Kijun
    # TK cross down: previous Tenkan > previous Kijun AND current Tenkan <= current Kijun
    tenkan_prev = np.roll(tenkan_aligned, 1)
    kijun_prev = np.roll(kijun_aligned, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan_prev < kijun_prev) & (tenkan_aligned >= kijun_aligned)
    tk_cross_down = (tenkan_prev > kijun_prev) & (tenkan_aligned <= kijun_aligned)
    
    # Calculate volume spike (20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 1 for roll, 20 for volume MA)
    start_idx = max(52, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_1d[i]) or
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku TK cross with 1d cloud filter and volume confirmation
        if position == 0:
            # Long: TK cross up AND price above 1d cloud AND volume spike
            if tk_cross_up[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below 1d cloud AND volume spike
            elif tk_cross_down[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down (regardless of cloud)
            if tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up (regardless of cloud)
            if tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0