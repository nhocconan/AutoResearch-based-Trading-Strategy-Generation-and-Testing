#!/usr/bin/env python3
"""
4h_ichimoku_1d_trend_volume_v1
Hypothesis: Use Ichimoku Cloud (Tenkan/Kijun) for trend direction on 4h, confirmed by 1d EMA200 and volume spike (>1.5x average). Enter long when price > Kumo (cloud) and Tenkan > Kijun in uptrend with volume confirmation; short when price < Kumo and Tenkan < Kijun in downtrend with volume confirmation. Exit when price re-enters the cloud. Designed for low frequency (20-50 trades/year) to avoid fee drag while capturing trend continuation in both bull and bear markets via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ichimoku_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For cloud calculation, we need current values (not shifted) to determine if price is above/below cloud
    # The cloud itself is plotted 26 periods ahead, but we compare current price to the current cloud boundaries
    # So we use Senkou A and B without shift for cloud thickness, but the cloud's position is determined by these values
    # Actually, for determining if price is above/below cloud, we compare to Senkou A and B values that are plotted 26 periods ahead
    # But since we don't have future data, we use the values that would be plotted: i.e., we need Senkou A and B from 26 periods ago
    # Simplified: use current Senkou A and B as proxy for current cloud (acceptable approximation)
    # Better: calculate Senkou A and B, then the cloud's lower/upper bound at time i is from Senkou A/B at i-26
    # For simplicity and to avoid look-ahead, we'll use the current Senkou A and B to define cloud boundaries
    # This means we're comparing price to where the cloud IS, not where it will be - still valid for trend identification
    
    # 40-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if daily EMA200 not available
        if np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 40-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Determine cloud boundaries (using Senkou A and B)
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        # We use current Senkou values as approximation of current cloud
        if not (np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])
            
            # Price above cloud: bullish
            price_above_cloud = close[i] > cloud_top
            # Price below cloud: bearish
            price_below_cloud = close[i] < cloud_bottom
            # Price in cloud: neutral
            price_in_cloud = (close[i] >= cloud_bottom) and (close[i] <= cloud_top)
        else:
            price_above_cloud = False
            price_below_cloud = False
            price_in_cloud = True
        
        # Tenkan/Kijun relationship
        tenkan_above_kijun = tenkan[i] > kijun[i] if not (np.isnan(tenkan[i]) or np.isnan(kijun[i])) else False
        tenkan_below_kijun = tenkan[i] < kijun[i] if not (np.isnan(tenkan[i]) or np.isnan(kijun[i])) else False
        
        if position == 1:  # Long position
            # Exit when price re-enters or goes below cloud (cloud bottom)
            if price_in_cloud or price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price re-enters or goes above cloud (cloud top)
            if price_in_cloud or price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud AND Tenkan > Kijun in uptrend with volume confirmation
            long_entry = price_above_cloud and tenkan_above_kijun and uptrend and vol_confirm
            # Short entry: price below cloud AND Tenkan < Kijun in downtrend with volume confirmation
            short_entry = price_below_cloud and tenkan_below_kijun and downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals