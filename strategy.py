#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku cloud breakout with daily Kumo twist filter and volume confirmation.
- Long when price breaks above Kumo cloud and Tenkan > Kijun (bullish TK cross), with daily Kumo twist bullish (Senkou A > Senkou B) and volume > 1.5x average.
- Short when price breaks below Kumo cloud and Tenkan < Kijun (bearish TK cross), with daily Kumo twist bearish (Senkou A < Senkou B) and volume > 1.5x average.
- Exit when price re-enters the Kumo cloud or TK cross reverses.
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross), while daily twist filters trend direction. Works in both bull/bear by requiring cloud alignment and twist confirmation.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Ichimoku and Kumo twist - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1-day Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo twist: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
    kumo_twist_bullish = senkou_a_1d > senkou_b_1d
    kumo_twist_bearish = senkou_a_1d < senkou_b_1d
    
    # Calculate 6-hour Ichimoku components for price/cloud relationship
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK cross signals
    tk_cross_bullish = tenkan_6h > kijun_6h
    tk_cross_bearish = tenkan_6h < kijun_6h
    
    # Price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Align HTF indicators to 6h timeframe
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        twist_bull = bool(kumo_twist_bullish_aligned[i])
        twist_bear = bool(kumo_twist_bearish_aligned[i])
        tk_bull = tk_cross_bullish[i]
        tk_bear = tk_cross_bearish[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price above cloud, bullish TK cross, bullish Kumo twist, volume confirmation
            if (price_above and tk_bull and twist_bull and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, bearish TK cross, bearish Kumo twist, volume confirmation
            elif (price_below and tk_bear and twist_bear and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price re-enters cloud OR TK cross turns bearish
                if not price_above or tk_bear:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price re-enters cloud OR TK cross turns bullish
                if not price_below or tk_bull:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_KumoTwist_Volume"
timeframe = "6h"
leverage = 1.0