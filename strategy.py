#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Filter_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku cloud twist (Senkou Span A/B cross) with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period median).
Enters long when price is above cloud, Tenkan > Kijun, and cloud is bullish (Senkou A > Senkou B) with volume confirmation and bullish 1d trend.
Enters short when price is below cloud, Tenkan < Kijun, and cloud is bearish (Senkou A < Senkou B) with volume confirmation and bearish 1d trend.
Exits when price crosses Tenkan-Kijun in opposite direction or cloud twist reverses.
Uses discrete position sizing (0.25) to minimize churn. Target: 75-150 trades over 4 years.
Works in both bull and bear markets by following 1d trend filter and requiring Ichimoku alignment + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud (Senkou Span A/B from 26 periods ago)
    senkou_a_lag26 = np.roll(senkou_a, 26)
    senkou_b_lag26 = np.roll(senkou_b, 26)
    senkou_a_lag26[:26] = np.nan
    senkou_b_lag26[:26] = np.nan
    
    # Cloud twist: Senkou A > Senkou B (bullish cloud) or Senkou A < Senkou B (bearish cloud)
    cloud_bullish = senkou_a_lag26 > senkou_b_lag26
    cloud_bearish = senkou_a_lag26 < senkou_b_lag26
    
    # Price above/below cloud
    price_above_cloud = (close > senkou_a_lag26) & (close > senkou_b_lag26)
    price_below_cloud = (close < senkou_a_lag26) & (close < senkou_b_lag26)
    
    # Tenkan/Kijun cross
    tenkan_above_kijun = tenkan > kijun
    tenkan_below_kijun = tenkan < kijun
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period Ichimoku, 50-period EMA)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(senkou_a_lag26[i]) or np.isnan(senkou_b_lag26[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price above cloud, Tenkan > Kijun, bullish cloud, volume confirm, bullish 1d trend
        if (price_above_cloud[i] and tenkan_above_kijun[i] and cloud_bullish[i] and 
            volume_confirm[i] and close[i] > ema50_1d_aligned[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price below cloud, Tenkan < Kijun, bearish cloud, volume confirm, bearish 1d trend
        elif (price_below_cloud[i] and tenkan_below_kijun[i] and cloud_bearish[i] and 
              volume_confirm[i] and close[i] < ema50_1d_aligned[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses Tenkan-Kijun in opposite direction OR cloud twist reverses
        elif position == 1 and (tenkan_below_kijun[i] or not cloud_bullish[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tenkan_above_kijun[i] or not cloud_bearish[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Filter_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0