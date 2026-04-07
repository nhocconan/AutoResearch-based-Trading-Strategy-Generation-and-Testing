#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Channel Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: Donchian breakouts capture momentum moves. Volume confirms institutional participation.
# ADX > 25 filters for trending markets, avoiding false breakouts in ranges.
# Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows).
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+,
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    if len(tr) >= tr_period:
        tr_smooth[tr_period-1] = np.nansum(tr[1:tr_period])
        dm_plus_smooth[tr_period-1] = np.sum(dm_plus[1:tr_period])
        dm_minus_smooth[tr_period-1] = np.sum(dm_minus[1:tr_period])
        
        # Wilder smoothing
        for i in range(tr_period, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX (smoothed DX)
    adx_period = 14
    adx = np.full_like(dx, np.nan)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.nanmean(dx[1:adx_period])
        for i in range(adx_period, len(dx)):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian Channel (20-period)
    donchian_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donchian_period-1, n):
        highest_high[i] = np.max(high[i-donchian_period+1:i+1])
        lowest_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(max(donchian_period-1, 30), n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band or volume drops
            if close[i] < lowest_low[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band or volume drops
            if close[i] > highest_high[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper band with volume confirmation
            if close[i] > highest_high[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band with volume confirmation
            elif close[i] < lowest_low[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals