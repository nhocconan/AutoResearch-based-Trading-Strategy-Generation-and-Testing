#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend
Hypothesis: Ichimoku TK cross with cloud filter and weekly trend alignment for 6h timeframe.
Works in bull markets via TK cross above cloud with weekly uptrend, and in bear markets via TK cross below cloud with weekly downtrend.
Cloud acts as dynamic support/resistance reducing false signals. Weekly trend filter ensures alignment with higher timeframe momentum.
Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
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
    
    # Weekly data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 trend filter (loaded ONCE)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily data for Ichimoku calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (standard settings: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = close_1d
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku calculation (52) + weekly EMA (200)
    start_idx = max(52, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # TK cross signals with cloud filter and weekly trend alignment
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i]
            tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i]
            
            # Price above/below cloud
            price_above_cloud = curr_close > cloud_top[i]
            price_below_cloud = curr_close < cloud_bottom[i]
            
            # Weekly trend alignment
            weekly_uptrend = curr_close > ema_200_1w_aligned[i]
            weekly_downtrend = curr_close < ema_200_1w_aligned[i]
            
            # Long entry: TK cross bullish + price above cloud + weekly uptrend + volume spike
            long_entry = tk_cross_bullish and price_above_cloud and weekly_uptrend and volume_spike[i]
            
            # Short entry: TK cross bearish + price below cloud + weekly downtrend + volume spike
            short_entry = tk_cross_bearish and price_below_cloud and weekly_downtrend and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TK cross turns bearish OR price closes below cloud bottom OR weekly trend turns down
            if (tenkan_aligned[i] < kijun_aligned[i]) or (curr_close < cloud_bottom[i]) or (curr_close < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross turns bullish OR price closes above cloud top OR weekly trend turns up
            if (tenkan_aligned[i] > kijun_aligned[i]) or (curr_close > cloud_top[i]) or (curr_close > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend"
timeframe = "6h"
leverage = 1.0