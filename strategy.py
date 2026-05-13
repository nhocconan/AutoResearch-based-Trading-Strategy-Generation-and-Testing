#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation. Uses 6h Tenkan/Kijun cross for entry timing, 1d cloud (Senkou Span A/B) for trend direction, and volume spike (>2x 20-bar avg) for confirmation. Designed for BTC/ETH robustness: Ichimoku provides dynamic support/resistance, cloud acts as trend filter, and volume confirms institutional participation. Targets 12-37 trades/year on 6h timeframe.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Ichimoku components
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
    
    # Calculate 1d trend filter: price above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Senkou B lookback
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and trend
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        is_bullish_cloud = senkou_a[i] > senkou_b[i]
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun, price above cloud, bullish cloud, price > 1d EMA50, volume spike
            tenkan_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            price_above_cloud = close[i] > cloud_top
            volume_spike = volume[i] > 2.0 * avg_volume[i]
            
            if (tenkan_cross_up and 
                price_above_cloud and 
                is_bullish_cloud and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun, price below cloud, bearish cloud, price < 1d EMA50, volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Tenkan cross down
                  close[i] < cloud_bottom and 
                  not is_bullish_cloud and  # bearish cloud
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price drops below cloud
            tenkan_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            price_below_cloud = close[i] < cloud_bottom
            
            if tenkan_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above cloud
            tenkan_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tenkan_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals