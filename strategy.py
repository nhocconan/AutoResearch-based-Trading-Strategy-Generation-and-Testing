#!/usr/bin/env python3

# 6h_ichimoku_cloud_trend_v1
# Hypothesis: Ichimoku-based trend-following strategy on 6h timeframe with 1d trend filter.
# Uses Ichimoku cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) to identify trend direction and momentum.
# Enters long when price is above cloud and Tenkan > Kijun, with 1d trend confirmation.
# Enters short when price is below cloud and Tenkan < Kijun, with 1d trend confirmation.
# Exits when price crosses back through Kijun-sen or cloud direction changes.
# Designed to work in both bull and bear markets by capturing strong trends while avoiding whipsaws in ranging markets.
# Target: 15-30 trades/year for low fee drift (60-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA50) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Cloud (Kumo) top and bottom
    # For cloud ahead, we shift Senkou spans by 26 periods
    senkou_a_leading = np.roll(senkou_a, 26)
    senkou_b_leading = np.roll(senkou_b, 26)
    # Fill leading values with NaN for first 26 periods
    senkou_a_leading[:26] = np.nan
    senkou_b_leading[:26] = np.nan
    
    # Cloud top is the higher of Senkou A and B
    cloud_top = np.where(~np.isnan(senkou_a_leading) & ~np.isnan(senkou_b_leading),
                         np.maximum(senkou_a_leading, senkou_b_leading), np.nan)
    # Cloud bottom is the lower of Senkou A and B
    cloud_bottom = np.where(~np.isnan(senkou_a_leading) & ~np.isnan(senkou_b_leading),
                            np.minimum(senkou_a_leading, senkou_b_leading), np.nan)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up (52 + 26 for Senkou shift)
    
    for i in range(start_idx, n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # Tenkan-Kijun relationship
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        if position == 1:  # Long position
            # Exit conditions: price drops below Kijun or cloud turns bearish (price below cloud)
            if close[i] < kijun[i] or not price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price rises above Kijun or cloud turns bullish (price above cloud)
            if close[i] > kijun[i] or not price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry conditions with daily trend filter
            # Long: price above cloud, Tenkan > Kijun, and daily uptrend
            if price_above_cloud and tenkan_above_kijun and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short: price below cloud, Tenkan < Kijun, and daily downtrend
            elif price_below_cloud and tenkan_below_kijun and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals