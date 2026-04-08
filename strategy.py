#!/usr/bin/env python3
# 1d_ichimoku_trend_weekly_v1
# Hypothesis: Daily Ichimoku Cloud trend with weekly ADX filter and volume confirmation.
# Long when: price > Kumo (cloud), Tenkan > Kijun, weekly ADX > 25, volume > 1.5x average.
# Short when: price < Kumo, Tenkan < Kijun, weekly ADX > 25, volume > 1.5x average.
# Exit when price crosses Kumo opposite direction or volume drops below average.
# Uses weekly trend filter to avoid false signals in ranging markets. Target: 10-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ichimoku_trend_weekly_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = ((high_tenkan + low_tenkan) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = ((high_kijun + low_kijun) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((high_senkou + low_senkou) / 2)
    
    # Kumo (Cloud): Senkou Span A and B shifted 26 periods ahead
    # For trend detection, we use current Senkou spans to represent cloud
    # In Ichimoku, cloud is plotted ahead, but for trend we compare price to current cloud components
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    period_adx = 14
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First period has no TR
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.full(len(close_1w), np.nan)
    dm_plus_smooth = np.full(len(close_1w), np.nan)
    dm_minus_smooth = np.full(len(close_1w), np.nan)
    
    # Initial average (first 14 periods)
    if len(tr) >= period_adx:
        atr[period_adx-1] = np.nanmean(tr[1:period_adx+1])
        dm_plus_smooth[period_adx-1] = np.nanmean(dm_plus[1:period_adx+1])
        dm_minus_smooth[period_adx-1] = np.nanmean(dm_minus[1:period_adx+1])
        
        # Wilder's smoothing
        for i in range(period_adx, len(close_1w)):
            atr[i] = (atr[i-1] * (period_adx-1) + tr[i]) / period_adx
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period_adx-1) + dm_plus[i]) / period_adx
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period_adx-1) + dm_minus[i]) / period_adx
    
    # Directional Indicators
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(period_adx-1, len(close_1w)):
        if not np.isnan(atr[i]) and atr[i] > 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # ADX: smoothed DX
    adx = np.full(len(close_1w), np.nan)
    if len(dx) >= 2 * period_adx - 1:
        adx[2*period_adx-2] = np.nanmean(dx[period_adx-1:2*period_adx-1])
        for i in range(2*period_adx-1, len(close_1w)):
            adx[i] = (adx[i-1] * (period_adx-1) + dx[i]) / period_adx
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period_kijun, vol_ma_period, 2*period_adx-1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Kumo (cloud) or volume drops below average
            if close[i] < kumo_top[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Kumo (cloud) or volume drops below average
            if close[i] > kumo_bottom[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Kumo, Tenkan > Kijun, weekly ADX > 25, volume surge
            if (close[i] > kumo_top[i] and 
                tenkan[i] > kijun[i] and 
                adx_aligned[i] > 25 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Kumo, Tenkan < Kijun, weekly ADX > 25, volume surge
            elif (close[i] < kumo_bottom[i] and 
                  tenkan[i] < kijun[i] and 
                  adx_aligned[i] > 25 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals