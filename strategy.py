#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
# Long when Tenkan-sen crosses above Kijun-sen, price above Kumo cloud, ADX > 25 (trending), and volume > 1.5x 20-period average.
# Short when Tenkan-sen crosses below Kijun-sen, price below Kumo cloud, ADX > 25, and volume > 1.5x 20-period average.
# Exit when Tenkan-sen crosses back in opposite direction of Kijun-sen.
# Uses Ichimoku for trend identification and momentum, ADX to filter ranging markets, volume to confirm breakouts.
# Target: 80-160 total trades over 4 years (20-40/year) for balanced frequency and edge.

name = "6h_Ichimoku_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo cloud boundaries (shifted 26 periods ahead)
    senkou_span_a_shifted = np.roll(senkou_span_a, 26)
    senkou_span_b_shifted = np.roll(senkou_span_b, 26)
    # Set first 26 values to NaN (no look-ahead)
    senkou_span_a_shifted[:26] = np.nan
    senkou_span_b_shifted[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    kumo_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed DM and ATR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    atr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Ichimoku and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Tenkan > Kijun, price above Kumo, ADX > 25, volume spike
            long_cond = (tenkan_sen[i] > kijun_sen[i]) and (tenkan_sen[i-1] <= kijun_sen[i-1]) and \
                        (close[i] > kumo_top[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: Tenkan < Kijun, price below Kumo, ADX > 25, volume spike
            short_cond = (tenkan_sen[i] < kijun_sen[i]) and (tenkan_sen[i-1] >= kijun_sen[i-1]) and \
                         (close[i] < kumo_bottom[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses back below Kijun
            if tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses back above Kijun
            if tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals