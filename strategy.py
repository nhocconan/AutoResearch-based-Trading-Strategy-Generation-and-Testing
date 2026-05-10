# %%
#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Follow
Hypothesis: Ichimoku cloud system on 6h with 1d trend filter (price above/below daily Kumo) and volume confirmation.
Works in both bull and bear markets by following higher timeframe trend and using cloud breakouts as high-probability entries.
Target: 15-25 trades/year per sensor with strict entry conditions to minimize fee drag.
"""

name = "6h_Ichimoku_Cloud_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52)
    tenkan_sen = np.full(n, np.nan)  # (9-period high + low)/2
    kijun_sen = np.full(n, np.nan)   # (26-period high + low)/2
    senkou_span_a = np.full(n, np.nan)  # (tenkan + kijun)/2 plotted 26 periods ahead
    senkou_span_b = np.full(n, np.nan)  # (52-period high + low)/2 plotted 26 periods ahead
    
    # Tenkan-sen (Conversion Line): 9-period
    for i in range(8, n):
        period_high = np.max(high[i-8:i+1])
        period_low = np.min(low[i-8:i+1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Kijun-sen (Base Line): 26-period
    for i in range(25, n):
        period_high = np.max(high[i-25:i+1])
        period_low = np.min(low[i-25:i+1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    for i in range(26, n):
        if not np.isnan(tenkan_sen[i-26]) and not np.isnan(kijun_sen[i-26]):
            senkou_span_a[i] = (tenkan_sen[i-26] + kijun_sen[i-26]) / 2
    
    # Senkou Span B (Leading Span B): 52-period high/low, plotted 26 periods ahead
    for i in range(51, n):
        period_high = np.max(high[i-51:i+1])
        period_low = np.min(low[i-51:i+1])
        senkou_span_b[i+26] = (period_high + period_low) / 2 if i+26 < n else np.nan
    
    # Calculate 1d Ichimoku cloud for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    tenkan_1d = np.full(len(close_1d), np.nan)
    for i in range(8, len(close_1d)):
        period_high = np.max(high_1d[i-8:i+1])
        period_low = np.min(low_1d[i-8:i+1])
        tenkan_1d[i] = (period_high + period_low) / 2
    
    # 1d Kijun-sen (26-period)
    kijun_1d = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        period_high = np.max(high_1d[i-25:i+1])
        period_low = np.min(low_1d[i-25:i+1])
        kijun_1d[i] = (period_high + period_low) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = np.full(len(close_1d), np.nan)
    for i in range(26, len(close_1d)):
        if not np.isnan(tenkan_1d[i-26]) and not np.isnan(kijun_1d[i-26]):
            senkou_a_1d[i] = (tenkan_1d[i-26] + kijun_1d[i-26]) / 2
    
    # 1d Senkou Span B (52-period)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        period_high = np.max(high_1d[i-51:i+1])
        period_low = np.min(low_1d[i-51:i+1])
        if i+26 < len(close_1d):
            senkou_b_1d[i+26] = (period_high + period_low) / 2
    
    # Calculate 1d Kumo (cloud) boundaries
    kumo_top_1d = np.full(len(close_1d), np.nan)   # max(Senkou A, Senkou B)
    kumo_bottom_1d = np.full(len(close_1d), np.nan) # min(Senkou A, Senkou B)
    for i in range(len(close_1d)):
        if not np.isnan(senkou_a_1d[i]) and not np.isnan(senkou_b_1d[i]):
            kumo_top_1d[i] = max(senkou_a_1d[i], senkou_b_1d[i])
            kumo_bottom_1d[i] = min(senkou_a_1d[i], senkou_b_1d[i])
    
    # Align 1d Kumo to 6h
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Calculate volume confirmation (volume > 1.5x 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(19, n):
        vol_sma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Need Senkou Span B calculated
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(vol_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # Determine cloud color and price position
        # Bullish cloud: Senkou Span A > Senkou Span B
        # Bearish cloud: Senkou Span A < Senkou Span B
        cloud_bullish = senkou_span_a[i] > senkou_span_b[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > senkou_span_a[i] and close[i] > senkou_span_b[i]
        price_below_cloud = close[i] < senkou_span_a[i] and close[i] < senkou_span_b[i]
        
        # 1d trend filter: price relative to 1d Kumo
        price_above_1d_kumo = not np.isnan(kumo_top_aligned[i]) and close[i] > kumo_top_aligned[i]
        price_below_1d_kumo = not np.isnan(kumo_bottom_aligned[i]) and close[i] < kumo_bottom_aligned[i]
        
        if position == 0:
            # Long: Price breaks above bullish cloud with TK cross bullish and 1d uptrend
            if (price_above_cloud and cloud_bullish and 
                tenkan_sen[i] > kijun_sen[i] and  # TK cross bullish
                price_above_1d_kumo and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bearish cloud with TK cross bearish and 1d downtrend
            elif (price_below_cloud and not cloud_bullish and 
                  tenkan_sen[i] < kijun_sen[i] and  # TK cross bearish
                  price_below_1d_kumo and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price closes below cloud or TK cross turns bearish
            if (close[i] < senkou_span_a[i] or close[i] < senkou_span_b[i] or
                tenkan_sen[i] < kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above cloud or TK cross turns bullish
            if (close[i] > senkou_span_a[i] or close[i] > senkou_span_b[i] or
                tenkan_sen[i] > kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# %%