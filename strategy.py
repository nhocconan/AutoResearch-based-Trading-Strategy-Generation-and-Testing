#!/usr/bin/env python3
"""
6h Ichimoku Cloud Filter with 1d Trend and Volume Confirmation
Hypothesis: Ichimoku cloud (Kumo) from daily timeframe acts as dynamic support/resistance.
Price above/below cloud with Tenkan-Kijun cross and volume confirmation captures
trend continuation while avoiding whipsaws. Works in bull/bear by using cloud as
filter and only trading in direction of cloud bias.
Target: 15-30 trades/year to minimize fee drag while capturing strong trends.
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
    
    # Get 1d data for Ichimoku and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise
    cloud_bullish = senkou_a_aligned > senkou_b_aligned
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for Ichimoku (52+26)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        is_bullish_cloud = cloud_bullish[i]
        vol_ok = vol_filter[i]
        
        # Determine if price is above/below cloud
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        
        if position == 0:
            # Long: price above bullish cloud, Tenkan > Kijun, volume confirmation
            if price_above_cloud and is_bullish_cloud and (tenkan_val > kijun_val) and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below bearish cloud, Tenkan < Kijun, volume confirmation
            elif price_below_cloud and (not is_bullish_cloud) and (tenkan_val < kijun_val) and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price drops below cloud bottom or Tenkan < Kijun
            if price < cloud_bottom_val or (tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price rises above cloud top or Tenkan > Kijun
            if price > cloud_top_val or (tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0