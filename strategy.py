#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX regime filter and volume confirmation.
Long when: Tenkan-sen crosses above Kijun-sen AND price above Kumo (cloud) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when: Tenkan-sen crosses below Kijun-sen AND price below Kumo AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Exit when: Tenkan-sen/Kijun-sen cross reverses OR price crosses opposite Kumo edge OR ADX < 20 (range).
Uses 1d HTF for ADX regime to avoid whipsaws in ranging markets. Ichimoku provides dynamic support/resistance via cloud.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Calculate 1d ADX for trend regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        # subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smooth(tr, period_adx)
    dm_plus_smooth = wilders_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilders_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smooth(dx, period_adx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Ichimoku calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Current Kumo (cloud) edges: Senkou Span A and B shifted back 26 periods
    # For point i, cloud is Senkou A[i-26] and Senkou B[i-26]
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_kijun, period_senkou_b, 26, 30)  # Kijun (26), Senkou B (52+26), Ichimoku cloud lag (26), ADX (30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate Tenkan/Kijun cross for entry signals
        if i >= start_idx + 1:
            tenkan_prev = tenkan[i-1]
            kijun_prev = kijun[i-1]
            tenkan_cross_above = tenkan_val > kijun_val and tenkan_prev <= kijun_prev
            tenkan_cross_below = tenkan_val < kijun_val and tenkan_prev >= kijun_prev
        else:
            tenkan_cross_above = False
            tenkan_cross_below = False
        
        # Determine cloud relationship
        price_above_kumo = price > kumo_top_val
        price_below_kumo = price < kumo_bottom_val
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above Kumo AND ADX > 25 (trending) AND volume spike
            if tenkan_cross_above and price_above_kumo and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below Kumo AND ADX > 25 AND volume spike
            elif tenkan_cross_below and price_below_kumo and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Tenkan crosses below Kijun OR price falls below Kumo bottom OR ADX < 20 (range)
                if tenkan_cross_below or price < kumo_bottom_val or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: Tenkan crosses above Kijun OR price rises above Kumo top OR ADX < 20
                if tenkan_cross_above or price > kumo_top_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0