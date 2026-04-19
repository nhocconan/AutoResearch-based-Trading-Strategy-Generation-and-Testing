#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day EMA34 trend filter and 4-hour Donchian breakout (10-period) with volume confirmation.
# Enhanced with ADX trend strength filter to reduce whipsaws in choppy markets.
# Target: 20-30 trades/year per symbol (~80-120 total over 4 years) to minimize fee drag.
# Works in bull markets via trend-following, avoids false signals in bear/chop via EMA34 and ADX filters.
name = "4h_1d_EMA34_ADX14_Donchian10_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for ADX14 trend strength (called ONCE before loop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_adx[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 4h data for Donchian10 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 10-period high/low
    high_10_4h = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10_4h = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    high_10_4h_aligned = align_htf_to_ltf(prices, df_4h, high_10_4h)
    low_10_4h_aligned = align_htf_to_ltf(prices, df_4h, low_10_4h)
    
    # Volume filter: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(high_10_4h_aligned[i]) or np.isnan(low_10_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 4h Donchian high with volume AND strong trend (ADX > 20)
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_10_4h_aligned[i] and 
                volume_filter[i] and 
                adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 4h Donchian low with volume AND strong trend (ADX > 20)
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_10_4h_aligned[i] and 
                  volume_filter[i] and 
                  adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 4h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 4h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals