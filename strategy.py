#!/usr/bin/env python3
"""
12h Ichimoku Cloud Breakout with Weekly Volume Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Breaking above/below cloud with volume confirmation captures strong momentum moves. Weekly volume filter ensures institutional participation, working in both bull (breakouts) and bear (breakdowns) markets. Targets 15-25 trades/year on 12h timeframe to minimize fee drag.
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
    
    # Get weekly data for volume filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly average volume
    weekly_vol_avg = pd.Series(df_weekly['volume'].values).rolling(window=4, min_periods=4).mean().values
    weekly_vol_avg_aligned = align_htf_to_ltf(prices, df_weekly, weekly_vol_avg)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # For signal, we use current close vs cloud ahead
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume filter: current volume > 1.5x weekly average volume
    vol_filter = volume > (weekly_vol_avg_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku (52 periods + buffer)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(weekly_vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        
        # Bullish: price above cloud (both spans)
        above_cloud = price > cloud_top[i] and price > cloud_bottom[i]
        # Bearish: price below cloud (both spans)
        below_cloud = price < cloud_top[i] and price < cloud_bottom[i]
        
        if position == 0:
            # Long: price breaks above cloud with volume
            if above_cloud and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with volume
            elif below_cloud and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price re-enters cloud
            if not above_cloud:  # price <= cloud_top or >= cloud_bottom (inside or below cloud)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price re-enters cloud
            if not below_cloud:  # price >= cloud_bottom or <= cloud_top (inside or above cloud)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Ichimoku_Cloud_Breakout_WeeklyVolume"
timeframe = "12h"
leverage = 1.0