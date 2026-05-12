#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== Ichimoku Components (1d) =====
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    period_senkou_b = 52
    max_high_senkou = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou + min_low_senkou) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    chikou = close_1d  # Will be aligned with offset
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou, additional_delay_bars=26)  # Lagging span needs delay
    
    # ===== Weekly Trend Filter =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    ma50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma50_1w)
    
    # ===== Daily Volume Spike Filter =====
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or
            np.isnan(senkou_b_6h[i]) or
            np.isnan(chikou_6h[i]) or
            np.isnan(ma50_1w_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + weekly uptrend + volume spike
            if (tenkan_6h[i] > kijun_6h[i] and  # TK cross bullish
                close[i] > cloud_top and         # Price above cloud
                close[i] > ma50_1w_aligned[i] and  # Weekly uptrend
                vol_spike_1d_aligned[i] > 0.5):   # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + weekly downtrend + volume spike
            elif (tenkan_6h[i] < kijun_6h[i] and  # TK cross bearish
                  close[i] < cloud_bottom and       # Price below cloud
                  close[i] < ma50_1w_aligned[i] and  # Weekly downtrend
                  vol_spike_1d_aligned[i] > 0.5):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tenkan_6h[i] < kijun_6h[i] or 
                close[i] < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tenkan_6h[i] > kijun_6h[i] or 
                close[i] > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals