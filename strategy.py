#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Kumo cloud AND Tenkan > Kijun AND 1w EMA50 rising AND volume > 1.5x MA20.
Short when price breaks below Kumo cloud AND Tenkan < Kijun AND 1w EMA50 falling AND volume > 1.5x MA20.
Exit when price re-enters Kumo cloud or Tenkan/Kijun cross reverses.
Ichimoku provides dynamic support/resistance, EMA50 filters higher timeframe trend, volume avoids low-momentum fakeouts.
Works in bull (trend-aligned breakouts) and bear (volume spikes on breakdowns with trend filter).
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Current Kumo cloud boundaries (use Senkou spans shifted back 26 periods to align with price)
    # Cloud at current point uses Senkou spans from 26 periods ago
    upper_cloud = np.maximum(np.roll(senkou_a, 26), np.roll(senkou_b, 26))
    lower_cloud = np.minimum(np.roll(senkou_a, 26), np.roll(senkou_b, 26))
    
    # For first 26 periods, cloud is not available
    upper_cloud[:26] = np.nan
    lower_cloud[:26] = np.nan
    
    # Calculate volume MA20 for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52 + 26, 20, 50)  # Ichimoku needs 52+26 for cloud, volume MA20, EMA50 1w
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Tenkan/Kijun cross for momentum
        if i >= start_idx + 1:
            tenkan_prev = tenkan[i-1]
            kijun_prev = kijun[i-1]
            tenkan_rising = tenkan_val > tenkan_prev and tenkan_val > kijun_val
            tenkan_falling = tenkan_val < tenkan_prev and tenkan_val < kijun_val
        else:
            tenkan_rising = tenkan_val > kijun_val
            tenkan_falling = tenkan_val < kijun_val
        
        # Volume filter: current volume > 1.5x MA20
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Price breaks above cloud AND Tenkan > Kijun AND 1w EMA50 rising AND volume filter
            if price > upper_cloud_val and tenkan_rising and ema_val > ema_50_aligned[max(i-1, start_idx)] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND Tenkan < Kijun AND 1w EMA50 falling AND volume filter
            elif price < lower_cloud_val and tenkan_falling and ema_val < ema_50_aligned[max(i-1, start_idx)] and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters cloud OR Tenkan crosses below Kijun
                if price < upper_cloud_val or tenkan_val < kijun_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters cloud OR Tenkan crosses above Kijun
                if price > lower_cloud_val or tenkan_val > kijun_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_IchimokuCloud_Breakout_1wEMA50_Trend_VolumeFilter"
timeframe = "6h"
leverage = 1.0