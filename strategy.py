#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ichimoku Cloud Filter with 1d Trend
# Hypothesis: Ichimoku cloud acts as dynamic support/resistance. 
# Price above/below cloud indicates trend direction, while TK cross provides entry signals.
# Using 1d cloud filter avoids counter-trend trades, improving performance in both bull and bear markets.
# Targets 15-30 trades/year with disciplined entries to avoid overtrading.

name = "6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d Ichimoku components (for cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = ((max_high_senkou + min_low_senkou) / 2)
    
    # Shift Senkou spans by 26 periods for cloud (already calculated above, now shift)
    senkou_a_shifted = np.roll(senkou_a_1d, 26)
    senkou_b_shifted = np.roll(senkou_b_1d, 26)
    # Set first 26 values to NaN due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # 6h TK Cross (Tenkan/Kijun cross)
    tk_diff = tenkan_6h - kijun_6h
    tk_diff_prev = np.roll(tk_diff, 1)
    tk_diff_prev[0] = np.nan
    tk_cross_up = (tk_diff > 0) & (tk_diff_prev <= 0)
    tk_cross_down = (tk_diff < 0) & (tk_diff_prev >= 0)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross down
            if close[i] < cloud_bottom[i] or tk_cross_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross up
            if close[i] > cloud_top[i] or tk_cross_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above cloud AND TK cross up
            if close[i] > cloud_top[i] and tk_cross_up[i]:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud AND TK cross down
            elif close[i] < cloud_bottom[i] and tk_cross_down[i]:
                position = -1
                signals[i] = -0.25
    
    return signals