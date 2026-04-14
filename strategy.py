#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1w Ichimoku Cloud (Kijun/Tenkan) with 1d volume spikes.
# Ichimoku provides trend direction and support/resistance (cloud) from weekly timeframe.
# Volume spikes on 1d confirm institutional interest for breakout/breakdown entries.
# Works in bull/bear markets: cloud acts as dynamic support/resistance, volume filters false breaks.
# Weekly Ichimoku avoids whipsaw; daily volume ensures momentum behind moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_sen = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe (wait for weekly bar close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a.values, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b.values, additional_delay_bars=26)
    
    # Load 1d data ONCE for volume spike
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume spike (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma  # Current volume vs 20-period average
    vol_ratio = np.where(vol_ma == 0, 1.0, vol_ratio)  # Avoid division by zero
    
    # Align volume ratio to 6h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 52 + 26)  # Ichimoku needs 52 periods + 26 shift
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        above_cloud = price > upper_cloud
        below_cloud = price < lower_cloud
        in_cloud = (price >= lower_cloud) & (price <= upper_cloud)
        
        # Trend: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        bullish_trend = tenkan_aligned[i] > kijun_aligned[i]
        bearish_trend = tenkan_aligned[i] < kijun_aligned[i]
        
        # Volume confirmation: significant spike (1.5x average)
        volume_spike = vol_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Enter long: price above cloud + bullish trend + volume spike
            if above_cloud and bullish_trend and volume_spike:
                position = 1
                signals[i] = position_size
            # Enter short: price below cloud + bearish trend + volume spike
            elif below_cloud and bearish_trend and volume_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud OR trend turns bearish
            if price < lower_cloud or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud OR trend turns bullish
            if price > upper_cloud or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wIchimoku_1dVolSpike_v1"
timeframe = "6h"
leverage = 1.0