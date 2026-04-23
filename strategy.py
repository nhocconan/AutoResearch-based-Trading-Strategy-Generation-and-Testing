#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
Long when price breaks above Kumo (cloud) AND Tenkan > Kijun AND 1d close > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when price breaks below Kumo (cloud) AND Tenkan < Kijun AND 1d close < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when price re-enters Kumo or Tenkan/Kijun cross reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, Ichimoku for dynamic support/resistance.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Ichimoku works in both bull and bear markets by adapting to volatility and trend strength.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For look-ahead safety, we use the cloud values from 26 periods ago
    senkou_a_lag26 = np.roll(senkou_a, 26)
    senkou_b_lag26 = np.roll(senkou_b, 26)
    senkou_a_lag26[:26] = np.nan
    senkou_b_lag26[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lag26, senkou_b_lag26)
    kumo_bottom = np.minimum(senkou_a_lag26, senkou_b_lag26)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 50, 20)  # Ichimoku needs 52, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Tenkan/Kijun cross for momentum
        if i >= start_idx + 1:
            tenkan_prev = tenkan[i-1]
            kijun_prev = kijun[i-1]
            tenkan_above_kijun = tenkan_val > kijun_val
            tenkan_below_kijun = tenkan_val < kijun_val
            tenkan_crossing_up = tenkan_prev <= kijun_prev and tenkan_val > kijun_val
            tenkan_crossing_down = tenkan_prev >= kijun_prev and tenkan_val < kijun_val
        else:
            tenkan_above_kijun = tenkan_val > kijun_val
            tenkan_below_kijun = tenkan_val < kijun_val
            tenkan_crossing_up = False
            tenkan_crossing_down = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Price above Kumo AND Tenkan > Kijun AND 1d EMA50 rising AND volume filter
            if (price > kumo_top_val and 
                tenkan_above_kijun and 
                ema_val > ema_50_aligned[max(0, i-1)] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below Kumo AND Tenkan < Kijun AND 1d EMA50 falling AND volume filter
            elif (price < kumo_bottom_val and 
                  tenkan_below_kijun and 
                  ema_val < ema_50_aligned[max(0, i-1)] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters Kumo OR Tenkan crosses below Kijun
                if (price < kumo_top_val or tenkan_crossing_down):
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters Kumo OR Tenkan crosses above Kijun
                if (price > kumo_bottom_val or tenkan_crossing_up):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_Breakout_1dEMA50_Trend_VolumeFilter"
timeframe = "6h"
leverage = 1.0