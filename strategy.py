#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with daily trend filter and volume confirmation.
Long when price > Kumo (cloud) from daily Ichimoku, Tenkan > Kijun, and volume > 1.5x average.
Short when price < Kumo, Tenkan < Kijun, and volume > 1.5x average.
Exit when price crosses Tenkan-Kijun line or volume drops below average.
Uses Ichimoku's forward-looking cloud for support/resistance, Tenkan/Kijun for momentum,
and volume for confirmation. Designed for 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        
        # Current daily values aligned to 6h
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_1d_current = vol_1d_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Enter long: price above cloud, Tenkan > Kijun, volume surge
            if (price_low > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and
                vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud, Tenkan < Kijun, volume surge
            elif (price_high < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses Tenkan-Kijun or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price below Tenkan-Kijun midpoint OR volume < average
                tenkan_kijun_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2.0
                if price_close < tenkan_kijun_mid:
                    exit_signal = True
                elif vol_1d_current < vol_ma_20_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price above Tenkan-Kijun midpoint OR volume < average
                tenkan_kijun_mid = (tenkan_aligned[i] + kijun_aligned[i]) / 2.0
                if price_close > tenkan_kijun_mid:
                    exit_signal = True
                elif vol_1d_current < vol_ma_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_IchimokuCloud_DailyTrend_Volume1.5x"
timeframe = "6h"
leverage = 1.0