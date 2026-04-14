# 4h_1d_Ichimoku_Cloud_Bullish_Bearish_Signal_With_Volume_Confirmation
# Hypothesis: Ichimoku cloud acts as dynamic support/resistance and trend filter.
# Long when price is above cloud AND Tenkan > Kijun (bullish cross) AND volume > 1.5x 20-period average.
# Short when price is below cloud AND Tenkan < Kijun (bearish cross) AND volume > 1.5x 20-period average.
# Exit when price crosses Tenkan-Kijun line or enters cloud.
# Uses 1d Ichimoku for trend filter (more stable) and 4h for entry timing.
# Target: 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    
    # Initialize arrays
    tenkan_1d = np.full_like(high_1d, np.nan)
    kijun_1d = np.full_like(high_1d, np.nan)
    senkou_span_a_1d = np.full_like(high_1d, np.nan)
    senkou_span_b_1d = np.full_like(high_1d, np.nan)
    
    # Calculate Tenkan-sen
    for i in range(period_tenkan - 1, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                        np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Calculate Kijun-sen
    for i in range(period_kijun - 1, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                       np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Calculate Senkou Span B
    for i in range(period_senkou_b - 1, len(high_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                               np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Senkou Span A is plotted 26 periods ahead
    # But we'll calculate it as (Tenkan + Kijun)/2 and align later
    for i in range(len(tenkan_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_span_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Load 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h data
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(period_senkou_b, period_kijun, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_ratio = volume_4h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: price outside cloud + TK cross + volume confirmation
            # Bullish: price above cloud AND Tenkan > Kijun AND volume > 1.5x average
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Bearish: price below cloud AND Tenkan < Kijun AND volume > 1.5x average
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price enters cloud OR Tenkan < Kijun (bearish cross)
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               tenkan_aligned[i] < kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price enters cloud OR Tenkan > Kijun (bullish cross)
            if (close[i] <= cloud_top and close[i] >= cloud_bottom) or \
               tenkan_aligned[i] > kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Ichimoku_Cloud_Bullish_Bearish_Signal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0