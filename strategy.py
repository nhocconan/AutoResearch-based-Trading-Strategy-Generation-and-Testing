#!/usr/bin/env python3
"""
6h Ichimoku Cloud Strategy with 1d Trend Filter
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction.
In bull markets, price stays above cloud; in bear markets, price stays below cloud.
TK cross (Tenkan/Kijun) signals momentum shifts. Using 1d cloud on 6s chart avoids
whipsaws and ensures we trade with higher timeframe trend. Cloud acts as dynamic
support/resistance, reducing false breakouts. This combination works in both
trending and ranging markets by filtering trades to higher timeframe direction.
Target: 20-30 trades/year to minimize fee drag while capturing sustained moves.
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
    
    # Get 1d data for Ichimoku cloud (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6s timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Determine if price is above or below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross signals
    tk_cross_up = tenkan_6h > kijun_6h  # Bullish cross
    tk_cross_down = tenkan_6h < kijun_6h  # Bearish cross
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for Ichimoku (52 + 26 shift)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_cloud = price_above_cloud[i]
        below_cloud = price_below_cloud[i]
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price above cloud + TK bullish cross + volume
            if above_cloud and tk_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish cross + volume
            elif below_cloud and tk_down and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price drops below cloud or TK turns bearish
            if below_cloud or tk_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price rises above cloud or TK turns bullish
            if above_cloud or tk_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_1dFilter"
timeframe = "6h"
leverage = 1.0