#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1
Hypothesis: 6h Ichimoku TK cross with 1d cloud filter and volume confirmation (>1.5x 20-period MA). 
Ichimoku provides trend, momentum, and support/resistance in one system. The cloud (Senkou Span) from 1d acts as a major trend filter - only take longs when price is above 1d cloud, shorts when below. TK cross (Tenkan/Kijun) provides timely entries on 6h. Volume confirmation ensures breakout strength. Designed to work in both bull and bear markets by requiring alignment with higher timeframe trend (cloud) and strong volume. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Ichimoku cloud (Senkou Span A/B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # The cloud is between Senkou Span A and B
    # We'll use the aligned versions for comparison with 6h price
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # === 6h Ichimoku TK Cross (Tenkan/Kijun) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (6h)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    # Kijun-sen (6h)
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # TK Cross signals
    tk_cross_above = tenkan_sen_6h > kijun_sen_6h  # Tenkan above Kijun = bullish
    tk_cross_below = tenkan_sen_6h < kijun_sen_6h  # Tenkan below Kijun = bearish
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        tenkan_6h = tenkan_sen_6h[i]
        kijun_6h = kijun_sen_6h[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above cloud, TK bullish cross, volume confirm
            long_condition = (price > cloud_top) and tk_cross_above[i] and volume_confirm
            # Short: price below cloud, TK bearish cross, volume confirm
            short_condition = (price < cloud_bottom) and tk_cross_below[i] and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price falls below cloud (trend change)
                elif price < cloud_top:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price rises above cloud (trend change)
                elif price > cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0