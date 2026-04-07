#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud filter with 1-week Tenkan/Kijun cross and volume confirmation
# Uses Ichimoku cloud (Senkou Span A/B) from 1d to filter trend direction
# Tenkan/Kijun cross from 1w for entry signals, volume > 1.5x average for confirmation
# Designed to work in both bull and bear markets via cloud filter (trend) and momentum cross
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ichimoku_cloud_1w_tk_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (shifted by 1 for completed bars)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 1-week data for Tenkan/Kijun cross (entry signal)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1w for entry signal
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (9-period) on 1w
    max_high_9w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    min_low_9w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (max_high_9w + min_low_9w) / 2
    
    # Kijun-sen (26-period) on 1w
    max_high_26w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    min_low_26w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (max_high_26w + min_low_26w) / 2
    
    # Align 1w Tenkan/Kijun to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    
    # Average volume for volume confirmation (24-period)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d cloud: price above/both spans = uptrend, below/both = downtrend
        price_above_cloud = (close[i] > senkou_a_1d_aligned[i]) and (close[i] > senkou_b_1d_aligned[i])
        price_below_cloud = (close[i] < senkou_a_1d_aligned[i]) and (close[i] < senkou_b_1d_aligned[i])
        
        # Volume confirmation: current volume > 1.5 * average volume
        volume_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Tenkan/Kijun cross from 1w: Tenkan crosses above Kijun = bullish, below = bearish
        tk_cross_bull = (tenkan_1w_aligned[i] > kijun_1w_aligned[i]) and (tenkan_1w_aligned[i-1] <= kijun_1w_aligned[i-1])
        tk_cross_bear = (tenkan_1w_aligned[i] < kijun_1w_aligned[i]) and (tenkan_1w_aligned[i-1] >= kijun_1w_aligned[i-1])
        
        # Long: price above cloud + bullish TK cross + volume confirmation
        if price_above_cloud and tk_cross_bull and volume_confirm:
            signals[i] = 0.25
        # Short: price below cloud + bearish TK cross + volume confirmation
        elif price_below_cloud and tk_cross_bear and volume_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals