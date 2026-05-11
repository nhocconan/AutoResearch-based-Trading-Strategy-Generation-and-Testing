# #!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_1dTrend_Volume
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter and support/resistance,
with Tenkan-Kijun cross on 6h for entry timing and volume confirmation. The Ichimoku cloud
provides dynamic support/resistance that adapts to volatility, working in both bull and bear
markets when aligned with daily trend. Tenkan-Kijun cross provides timely entries while
cloud acts as dynamic stop/reversal level.
"""

name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Ichimoku (for trend and cloud) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    tenkan_d, kijun_d, senkou_a_d, senkou_b_d, chikou_d = calculate_ichimoku(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_d)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_d)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_d)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_d)
    
    # Cloud top and bottom (Senkou A and B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === 6h Tenkan-Kijun Cross (for entry timing) ===
    tenkan_6h_raw = pd.Series(high).rolling(window=9, min_periods=9).max()
    tenkan_6h_raw = (tenkan_6h_raw + pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h_raw = pd.Series(high).rolling(window=26, min_periods=26).max()
    kijun_6h_raw = (kijun_6h_raw + pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    tenkan_6h_raw = tenkan_6h_raw.values
    kijun_6h_raw = kijun_6h_raw.values
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Ichimoku calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_6h_raw[i]) or np.isnan(kijun_6h_raw[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun, price above cloud, with volume
            tk_cross_up = (tenkan_6h_raw[i] > kijun_6h_raw[i] and 
                          tenkan_6h_raw[i-1] <= kijun_6h_raw[i-1])
            if (tk_cross_up and price_above_cloud and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun, price below cloud, with volume
            tk_cross_down = (tenkan_6h_raw[i] < kijun_6h_raw[i] and 
                            tenkan_6h_raw[i-1] >= kijun_6h_raw[i-1])
            if (tk_cross_down and price_below_cloud and volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price falls below cloud
            tk_cross_down = (tenkan_6h_raw[i] < kijun_6h_raw[i] and 
                            tenkan_6h_raw[i-1] >= kijun_6h_raw[i-1])
            price_below_cloud = close[i] < cloud_top[i]  # Exit if below cloud top
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            tk_cross_up = (tenkan_6h_raw[i] > kijun_6h_raw[i] and 
                          tenkan_6h_raw[i-1] <= kijun_6h_raw[i-1])
            price_above_cloud = close[i] > cloud_bottom[i]  # Exit if above cloud bottom
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals