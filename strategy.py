# 6h_Ichimoku_Cloud_Breakout
# Hypothesis: Ichimoku Cloud acts as dynamic support/resistance. In trending markets, price breaks above/below the cloud with TK cross confirmation signal strong moves. Using 1d Ichimoku for trend filter reduces whipsaws. Works in bull (breakouts up) and bear (breakdowns down) by following cloud color and TK cross. Low trade frequency expected due to strict cloud breakout + TK cross + volume confirmation.

#!/usr/bin/env python3

name = "6h_Ichimoku_Cloud_Breakout"
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
    volume = prices['volume'].values
    
    # 1d Ichimoku components for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_1d = (max_high_10 + min_low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b_1d = ((max_high_52 + min_low_52) / 2)
    
    # Align 1d Ichimoku to 6h timeframe (wait for 1d bar to close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # 6m Ichimoku for entry signals (TK cross)
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_9 = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max()
    min_low_9 = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min()
    tenkan_6h = (max_high_9 + min_low_9) / 2
    
    max_high_26 = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max()
    min_low_26 = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min()
    kijun_6h = (max_high_26 + min_low_26) / 2
    
    # Volume confirmation (24-period average = 4 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 24) + 5  # Need enough history
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or \
           np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross on 6h
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above cloud, TK bullish, volume confirmation
            if close[i] > cloud_top and tk_cross_bullish and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud, TK bearish, volume confirmation
            elif close[i] < cloud_bottom and tk_cross_bearish and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud base OR TK turns bearish
            if close[i] < cloud_bottom or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud top OR TK turns bullish
            if close[i] > cloud_top or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals