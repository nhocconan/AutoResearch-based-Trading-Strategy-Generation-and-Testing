#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Vortex_Trend_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Vortex Indicator (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Vortex calculations
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vi_plus = np.concatenate([[np.nan], vm_plus])
    vi_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_plus_sum = pd.Series(vi_plus).rolling(window=14, min_periods=14).sum().values
    vi_minus_sum = pd.Series(vi_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus_final = vi_plus_sum / tr_sum
    vi_minus_final = vi_minus_sum / tr_sum
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 12h
    vi_plus_12h = align_htf_to_ltf(prices, df_1d, vi_plus_final)
    vi_minus_12h = align_htf_to_ltf(prices, df_1d, vi_minus_final)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_12h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus_12h[i]) or np.isnan(vi_minus_12h[i]) or
            np.isnan(ema50_1d_12h[i]) or np.isnan(volume_filter_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vi_plus_val = vi_plus_12h[i]
        vi_minus_val = vi_minus_12h[i]
        trend = ema50_1d_12h[i]
        vol_filter = volume_filter_12h[i]
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish vortex) with volume and above trend
            if vi_plus_val > vi_minus_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish vortex) with volume and below trend
            elif vi_minus_val > vi_plus_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: vortex turns bearish (VI- > VI+)
            if vi_minus_val > vi_plus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: vortex turns bullish (VI+ > VI-)
            if vi_plus_val > vi_minus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals