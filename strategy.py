#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_v1
# Strategy: 6h Ichimoku Cloud with 1d cloud filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud identifies strong support/resistance and trend direction.
# The 1d cloud acts as a higher-timeframe filter to avoid counter-trend trades.
# Volume > 1.3x 20-period average confirms institutional participation.
# Designed for low trade frequency (~15-30/year) to minimize fee drift.
# Works in bull markets via long entries above cloud and bear markets via short entries below cloud.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen = ((high_9 + low_9) / 2).values
    
    # 6h Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen = ((high_26 + low_26) / 2).values
    
    # 6h Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # 6h Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # 1d Ichimoku for higher timeframe filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen_1d = ((high_9_1d + low_9_1d) / 2).values
    
    # 1d Kijun-sen (26-period)
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen_1d = ((high_26_1d + low_26_1d) / 2).values
    
    # 1d Senkou Span A
    senkou_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    # 1d Senkou Span B
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after 52-period lookback
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 6h cloud boundaries (future values already aligned)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        # Determine 1d cloud boundaries
        upper_cloud_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Ichimoku signals
        tk_cross_bullish = tenkan_sen[i] > kijun_sen[i]
        tk_cross_bearish = tenkan_sen[i] < kijun_sen[i]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Higher timeframe filter: price relative to 1d cloud
        price_above_1d_cloud = close[i] > upper_cloud_1d
        price_below_1d_cloud = close[i] < lower_cloud_1d
        
        # Entry conditions
        # Long: Price above 6h cloud AND TK bullish cross AND price above 1d cloud AND volume confirmation
        if (price_above_cloud and tk_cross_bullish and price_above_1d_cloud and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price below 6h cloud AND TK bearish cross AND price below 1d cloud AND volume confirmation
        elif (price_below_cloud and tk_cross_bearish and price_below_1d_cloud and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite cloud boundary (opposite color cloud)
        elif position == 1 and price_below_cloud:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_cloud:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals