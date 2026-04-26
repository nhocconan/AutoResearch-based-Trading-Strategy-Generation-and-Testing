#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend
Hypothesis: 6h Ichimoku TK cross with 1d cloud filter and volume confirmation. Works in bull/bear markets by using the cloud as dynamic support/resistance and TK cross for momentum, with 1d trend filter to avoid counter-trend trades. Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components (no additional delay needed as they are calculated from completed bars)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Using 1d index for alignment, but values are from 6h
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (max of all periods)
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 50, 20) + 26  # +26 for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud direction (green = bullish, red = bearish)
        # Cloud is bullish when Senkou A > Senkou B
        cloud_bullish = senkou_a_aligned[i] > senkou_b_aligned[i]
        
        # Long logic: TK cross bullish (Tenkan > Kijun) + price above cloud + price > 1d EMA50 (uptrend) + volume spike
        if (tenkan_aligned[i] > kijun_aligned[i] and  # TK cross bullish
            close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i] and  # Price above cloud
            close[i] > ema_50_1d_aligned[i] and  # Price above 1d EMA50
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross bearish (Tenkan < Kijun) + price below cloud + price < 1d EMA50 (downtrend) + volume spike
        elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK cross bearish
              close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i] and  # Price below cloud
              close[i] < ema_50_1d_aligned[i] and  # Price below 1d EMA50
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross reverses or price crosses cloud in opposite direction
        elif position == 1 and (tenkan_aligned[i] < kijun_aligned[i] or  # TK cross bearish
                                close[i] < senkou_a_aligned[i] or close[i] < senkou_b_aligned[i]):  # Price below cloud
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tenkan_aligned[i] > kijun_aligned[i] or  # TK cross bullish
                                 close[i] > senkou_a_aligned[i] or close[i] > senkou_b_aligned[i]):  # Price above cloud
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0