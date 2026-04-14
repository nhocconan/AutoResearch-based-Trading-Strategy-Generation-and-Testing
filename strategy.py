#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Ichimoku Cloud for trend filter and 1-day Williams %R for mean-reversion entries.
# Weekly Ichimoku Cloud (Tenkan/Kijun) determines primary trend direction - only trade long when price above cloud, short when below.
# Daily Williams %R (14-period) provides oversold/overbought signals for counter-trend entries within the weekly trend.
# Volume confirmation (>1.5x 24-period average) filters false signals.
# Designed to work in both bull and bear markets by using weekly trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 9:
        return np.zeros(n)
    
    # Calculate Ichimoku Cloud on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_high = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    tenkan_low = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_high = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    kijun_low = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b_high = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    senkou_b_low = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((senkou_b_high + senkou_b_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Load 1d data ONCE for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close)/(Highest High - Lowest Low) * -100
    period_williams = 14
    highest_high = pd.Series(high_1d).rolling(window=period_williams, min_periods=period_williams).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period_williams, min_periods=period_williams).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(26, 24)  # Need Ichimoku and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below weekly cloud
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for entries based on Williams %R extremes within weekly trend
            # Long: Williams %R oversold (< -80) AND price above weekly cloud
            if (williams_r_aligned[i] < -80 and 
                price_above_cloud and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND price below weekly cloud
            elif (williams_r_aligned[i] > -20 and 
                  price_below_cloud and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or price breaks below cloud
            if (williams_r_aligned[i] > -50 or 
                close[i] < cloud_bottom):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or price breaks above cloud
            if (williams_r_aligned[i] < -50 or 
                close[i] > cloud_top):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wIchimoku_1dWilliamsR_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0