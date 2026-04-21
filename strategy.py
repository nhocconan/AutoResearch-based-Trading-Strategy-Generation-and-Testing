#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_WeeklyTrend_12hVolume_v1
Hypothesis: 6h Ichimoku cloud breakout filtered by 1w EMA200 trend and 12h volume spike (2.0x average).
Long when Tenkan > Kijun and price above cloud and above 1w EMA200; short when Tenkan < Kijun and price below cloud and below 1w EMA200.
Volume confirmation reduces false breakouts. Uses discrete sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets via weekly trend alignment and strict entry filters.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA200 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA200 for trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 6h Ichimoku Cloud components (Tenkan, Kijun, Senkou Span A/B) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Current cloud boundaries: Senkou Span A and B from 26 periods ago
    senkou_a_lag = np.roll(senkou_a, 26)
    senkou_b_lag = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN (not available)
    senkou_a_lag[:26] = np.nan
    senkou_b_lag[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # === 12h Volume confirmation (50-period average) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup for Ichimoku calculations
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])
            or np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = prices['volume'].values[i]
        tenkan_now = tenkan[i]
        kijun_now = kijun[i]
        cloud_top_now = cloud_top[i]
        cloud_bottom_now = cloud_bottom[i]
        ema_trend = ema_200_1w_aligned[i]
        vol_avg = vol_ma_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        # Price above/below cloud
        price_above_cloud = price > cloud_top_now
        price_below_cloud = price < cloud_bottom_now
        
        if position == 0:
            # Long: Tenkan > Kijun, price above cloud, above weekly EMA200, volume confirmed
            long_condition = (tenkan_now > kijun_now) and price_above_cloud and (price > ema_trend) and volume_confirmed
            # Short: Tenkan < Kijun, price below cloud, below weekly EMA200, volume confirmed
            short_condition = (tenkan_now < kijun_now) and price_below_cloud and (price < ema_trend) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            # Trend reversal: Tenkan < Kijun
            if tenkan_now < kijun_now:
                signals[i] = 0.0
                position = 0
            # Price falls below cloud
            elif price < cloud_top_now:
                signals[i] = 0.0
                position = 0
            # Weekly trend reversal: price below EMA200
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            # Trend reversal: Tenkan > Kijun
            if tenkan_now > kijun_now:
                signals[i] = 0.0
                position = 0
            # Price rises above cloud
            elif price > cloud_bottom_now:
                signals[i] = 0.0
                position = 0
            # Weekly trend reversal: price above EMA200
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_12hVolume_v1"
timeframe = "6h"
leverage = 1.0