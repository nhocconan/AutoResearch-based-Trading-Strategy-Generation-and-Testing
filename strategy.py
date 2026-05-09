#!/usr/bin/env python3
# Hypothesis: 1h Ichimoku Cloud with 4h ADX trend filter and volume confirmation
# Long when price above Kumo, Tenkan > Kijun, ADX > 25, volume > 1.5x average
# Short when price below Kumo, Tenkan < Kijun, ADX > 25, volume > 1.5x average
# Exit when price crosses Tenkan-Kijun line or Kumo
# Uses multi-timeframe trend confirmation to filter noise, targeting 20-40 trades/year
# Designed to work in both bull (trend following) and bear (counter-trend at extremes) markets

name = "1h_Ichimoku_ADX_Volume_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Load 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = pd.Series(high_4h).diff()
    tr2 = pd.Series(low_4h).diff().abs()
    tr3 = pd.Series(close_4h).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high_4h).diff()
    dm_minus = -pd.Series(low_4h).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align Ichimoku and ADX to 1h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_b.values)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_kumo = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Enter long: price above Kumo, Tenkan > Kijun, ADX > 25, volume spike
            if (close[i] > upper_kumo and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price below Kumo, Tenkan < Kijun, ADX > 25, volume spike
            elif (close[i] < lower_kumo and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Kumo or Tenkan-Kijun cross down
            if (close[i] < upper_kumo) or (tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above Kumo or Tenkan-Kijun cross up
            if (close[i] > lower_kumo) or (tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals