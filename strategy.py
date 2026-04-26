#!/usr/bin/env python3
"""
6h_IchiKumo_Twist_1dTrend_Regime_v1
Hypothesis: 6h Ichimoku TK cross with Kumo twist filter (price breaks Kumo in direction of TK cross) + 1d trend filter (EMA50) + volume confirmation.
Only trade when price is above/below Kumo and TK cross aligns with 1d trend. Uses volume spike to avoid false breakouts.
Designed for 6h timeframe to work in both bull and bear markets via 1d trend filter and Kumo twist as momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) by requiring confluence of TK cross, Kumo twist, trend, and volume.
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
    
    # Load 1d data ONCE before loop for HTF trend and Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    upper_kumo = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_kumo = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross = np.where(tenkan_aligned > kijun_aligned, 1, -1)  # 1 = bullish cross, -1 = bearish cross
    
    # Kumo twist: Senkou Span A crossing above/below Senkou Span B (future cloud twist)
    # We use current alignment to detect twist - when Senkou A > Senkou B = bullish twist
    kumo_twist = np.where(senkou_a_aligned > senkou_b_aligned, 1, -1)  # 1 = bullish twist, -1 = bearish twist
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 52 for Senkou B, 20 for volume MA)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(upper_kumo[i]) or 
            np.isnan(lower_kumo[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Kumo twist condition: price must be above/below Kumo in direction of twist
        price_above_kumo = close[i] > upper_kumo[i]
        price_below_kumo = close[i] < lower_kumo[i]
        
        # Bullish conditions: price above Kumo + bullish TK cross + bullish Kumo twist + uptrend + volume
        if (price_above_kumo and tk_cross[i] == 1 and kumo_twist[i] == 1 and 
            htf_trend[i] == 1 and volume_spike):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Bearish conditions: price below Kumo + bearish TK cross + bearish Kumo twist + downtrend + volume
        elif (price_below_kumo and tk_cross[i] == -1 and kumo_twist[i] == -1 and 
              htf_trend[i] == -1 and volume_spike):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchiKumo_Twist_1dTrend_Regime_v1"
timeframe = "6h"
leverage = 1.0