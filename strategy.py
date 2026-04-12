#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_ichimoku_cloud_trend
# Uses weekly Ichimoku cloud (tenkan-sen, kijun-sen, senkou span A/B) to determine trend direction.
# Enters long when price is above cloud and tenkan > kijun (bullish TK cross).
# Enters short when price is below cloud and tenkan < kijun (bearish TK cross).
# Uses daily volume confirmation to filter weak breakouts.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (trend following shorts).

name = "6h_1w_1d_ichimoku_cloud_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # need at least 1 year of weekly data
        return np.zeros(n)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Ichimoku components (weekly)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Daily volume confirmation: volume > 1.5 * 20-day average
    vol_ma_daily = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_confirm_daily = df_1d['volume'].values > (vol_ma_daily * 1.5)
    vol_confirm_6h = align_htf_to_ltf(prices, df_1d, vol_confirm_daily.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if Ichimoku not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Check TK cross
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Check price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Require volume confirmation
        if vol_confirm_6h[i] < 0.5:  # low volume
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price above cloud + bullish TK cross
        if price_above_cloud and tk_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below cloud + bearish TK cross
        elif price_below_cloud and tk_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite conditions
        elif (price_below_cloud or not tk_bullish) and position == 1:
            position = 0
            signals[i] = 0.0
        elif (price_above_cloud or not tk_bearish) and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals