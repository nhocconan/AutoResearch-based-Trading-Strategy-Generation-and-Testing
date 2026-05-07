#!/usr/bin/env python3
name = "6h_ADX_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # ADX(14) on 6h
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed ATR, DM+ and DM- (Wilder's smoothing)
    atr = np.full(n, np.nan)
    dm_plus_smooth = np.full(n, np.nan)
    dm_minus_smooth = np.full(n, np.nan)
    
    # Initial values
    atr[13] = np.nansum(tr[1:14])
    dm_plus_smooth[13] = np.nansum(dm_plus[1:14])
    dm_minus_smooth[13] = np.nansum(dm_minus[1:14])
    
    # Wilder's smoothing
    for i in range(14, n):
        atr[i] = atr[i-1] - (atr[i-1] / 14) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / 14) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    adx = np.full(n, np.nan)
    adx[27] = np.nanmean(dx[14:28])  # First ADX value after smoothing
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for Ichimoku and ADX
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross up, price above cloud, ADX > 20
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > cloud_top and 
                adx[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud, ADX > 20
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom and 
                  adx[i] > 20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross down or price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with cloud filter and ADX trend strength filter.
# Ichimoku provides support/resistance via cloud and momentum via TK cross.
# ADX > 20 ensures we only trade when trend is strong enough to avoid whipsaw.
# Works in both bull and bear markets by following the trend direction.
# Position size 0.25 limits drawdown. Target: 15-25 trades/year.