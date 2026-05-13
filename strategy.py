#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Filter
Hypothesis: Ichimoku system (Tenkan/Kijun cross + cloud color) with 1d trend filter works in both bull and bear markets.
Long: Tenkan > Kijun + price above cloud + 1d uptrend.
Short: Tenkan < Kijun + price below cloud + 1d downtrend.
Exit on opposite Tenkan/Kijun cross or trend flip.
Uses Ichimoku parameters (9,26,52) on 6h with 1d trend filter.
Target: 20-50 trades/year per symbol.
"""

name = "6h_Ichimoku_Cloud_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
    # Tenkan-sen: (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # For cloud calculation, we need to shift Senkou spans forward by 26 periods
    # But for cloud color at current point, we use Senkou values from 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top/bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise
    cloud_green = senkou_a_shifted > senkou_b_shifted
    cloud_red = senkou_a_shifted < senkou_b_shifted
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any NaN values in Ichimoku components
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            signals[i] = 0.0
            continue
            
        tk_cross_up = tenkan[i] > kijun[i]
        tk_cross_down = tenkan[i] < kijun[i]
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: Tenkan > Kijun + price above cloud + green cloud + 1d uptrend
            if tk_cross_up and price_above_cloud and cloud_green[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan < Kijun + price below cloud + red cloud + 1d downtrend
            elif tk_cross_down and price_below_cloud and cloud_red[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan < Kijun or price below cloud or 1d trend turns down
            if tk_cross_down or price_below_cloud or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan > Kijun or price above cloud or 1d trend turns up
            if tk_cross_up or price_above_cloud or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals