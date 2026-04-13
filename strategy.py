#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w ADX regime filter
    # Long: price breaks above Camarilla H3 AND 1d volume > 1.5x avg AND 1w ADX > 25 (trending)
    # Short: price breaks below Camarilla L3 AND 1d volume > 1.5x avg AND 1w ADX > 25 (trending)
    # Exit: price returns to Camarilla Pivot level or ADX < 20 (range)
    # Using 12h timeframe for low trade frequency, Camarilla for structure,
    # 1d volume for confirmation, 1w ADX for regime filter (avoid chop).
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from daily OHLC
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1 / 4)
    # L3 = pivot - (range * 1.1 / 4)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    H3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    L3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 1d volume MA (20-period) for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate weekly ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 12h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx_1w_aligned[i] > 25
        ranging = adx_1w_aligned[i] < 20  # exit condition
        
        # Volume confirmation
        vol_confirm = volume_spike_1d_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > H3_1d_aligned[i]
        short_breakout = close[i] < L3_1d_aligned[i]
        
        # Return to pivot (exit condition)
        near_pivot = np.abs(close[i] - pivot_1d_aligned[i]) < (0.1 * np.abs(H3_1d_aligned[i] - L3_1d_aligned[i]))
        
        # Entry logic: breakout + volume + trending regime
        long_entry = long_breakout and vol_confirm and trending
        short_entry = short_breakout and vol_confirm and trending
        
        # Exit logic: return to pivot OR ranging market
        long_exit = near_pivot or ranging
        short_exit = near_pivot or ranging
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_camarilla_breakout_volume_adx_v2"
timeframe = "12h"
leverage = 1.0