#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels and 1-day Ichimoku cloud for trend direction.
# Weekly pivot levels provide strong support/resistance from institutional activity.
# Ichimoku cloud from daily timeframe filters trades to align with higher timeframe trend.
# Long when price breaks above weekly R1 with price above Ichimoku cloud (bullish).
# Short when price breaks below weekly S1 with price below Ichimoku cloud (bearish).
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Designed to work in both bull and bear markets by using weekly pivot structure and
# daily trend filter to avoid counter-trend trades.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for each week (using previous week's data)
    pivot = np.full(len(df_1w), np.nan)
    r1 = np.full(len(df_1w), np.nan)
    s1 = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        # Use previous week's data to calculate current week's pivot
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        p = (ph + pl + pc) / 3.0
        r1_val = 2 * p - pl
        s1_val = 2 * p - ph
        
        pivot[i] = p
        r1[i] = r1_val
        s1[i] = s1_val
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Load daily data ONCE for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    # Senkou Span B needs 26-period displacement (already built into calculation)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # The cloud is between Senkou Span A and B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Price above cloud: bullish
    # Price below cloud: bearish
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(26, 20)  # Need Ichimoku and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above weekly R1 or below weekly S1
            # Only trade in direction of Ichimoku cloud (trend filter)
            
            # Long: price breaks above weekly R1 AND price above Ichimoku cloud (bullish)
            if (close[i] > r1_aligned[i] and 
                price_above_cloud[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S1 AND price below Ichimoku cloud (bearish)
            elif (close[i] < s1_aligned[i] and 
                  price_below_cloud[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or breaks below cloud bottom
            if (close[i] <= pivot_aligned[i] or 
                close[i] < cloud_bottom[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or breaks above cloud top
            if (close[i] >= pivot_aligned[i] or 
                close[i] > cloud_top[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_1dIchimoku_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0