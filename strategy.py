#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_WeeklyTrend_12hVolume_v2
Hypothesis: 6h Ichimoku cloud (TK cross + cloud filter) aligned with 1w trend and confirmed by 12h volume spike.
Ichimoku provides multi-component trend/filter system: Tenkan/Kijun cross for momentum,
Senkou Span A/B for dynamic support/resistance (cloud). Weekly trend filter ensures alignment
with higher timeframe direction. Volume spike confirms institutional participation.
Designed to work in both bull and bear markets via weekly trend alignment and cloud filter.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend, 12h for volume confirmation)
    df_1w = get_htf_data(prices, '1w')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1w) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # === Ichimoku components (9,26,52) on 6h close ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # Cloud (Kumo): between Senkou Span A and B
    # Bullish when price > cloud and Senkou A > Senkou B
    # Bearish when price < cloud and Senkou A < Senkou B
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # === 1w trend filter (EMA50) ===
    df_1w_close = df_1w['close'].values
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h volume confirmation (20-period average) ===
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for Ichimoku (52+26)
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) 
            or np.isnan(cloud_bottom[i]) or np.isnan(ema_50_1w_aligned[i]) 
            or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = prices['volume'].iloc[i] if hasattr(prices['volume'], 'iloc') else prices['volume'][i]
        vol_avg = vol_ma_12h_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 12h average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        # Cloud bullish/bearish
        cloud_bullish = senkou_a[i] > senkou_b[i]
        cloud_bearish = senkou_a[i] < senkou_b[i]
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, weekly uptrend, volume confirmed
            tk_bullish = tenkan[i] > kijun[i]
            long_condition = tk_bullish and price_above_cloud and cloud_bullish and (price > ema_trend) and volume_confirmed
            
            # Short: TK cross bearish, price below cloud, weekly downtrend, volume confirmed
            tk_bearish = tenchan[i] < kijun[i] if 'tenchan' in locals() else tenkan[i] < kijun[i]
            short_condition = tk_bearish and price_below_cloud and cloud_bearish and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on TK cross bearish or price breaks below cloud
            tk_bearish_exit = tenkan[i] < kijun[i]
            if tk_bearish_exit or price < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on TK cross bullish or price breaks above cloud
            tk_bullish_exit = tenkan[i] > kijun[i]
            if tk_bullish_exit or price > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_12hVolume_v2"
timeframe = "6h"
leverage = 1.0