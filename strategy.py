#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout with daily volume confirmation and 12h ADX trend filter
# Hypothesis: Donchian breakouts capture momentum; volume confirms institutional participation; ADX ensures trending markets.
# Works in bull via upward breakouts, in bear via downward breakdowns. Filters reduce false signals.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_donchian20_vol_adx_v1"
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
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    # Smooth TR, DM+
    tr_period = 14
    tr_smooth = np.concatenate([[np.mean(tr_12h[:tr_period])], np.zeros(len(tr_12h) - tr_period)])
    for i in range(tr_period, len(tr_12h)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr_12h[i]
    
    dm_plus_smooth = np.concatenate([[np.mean(dm_plus[:tr_period])], np.zeros(len(dm_plus) - tr_period)])
    for i in range(tr_period, len(dm_plus)):
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
    
    dm_minus_smooth = np.concatenate([[np.mean(dm_minus[:tr_period])], np.zeros(len(dm_minus) - tr_period)])
    for i in range(tr_period, len(dm_minus)):
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(tr_12h)
    mask = (di_plus + di_minus) != 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
    
    adx = np.concatenate([[np.nan] * tr_period, np.zeros(len(dx) - tr_period)])
    for i in range(tr_period, len(dx)):
        if i == tr_period:
            adx[i] = np.mean(dx[:i+1])
        else:
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (20-period low)
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (20-period high)
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian upper band + volume confirmation + trend filter
            if close[i] > highest_high[i] and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band + volume confirmation + trend filter
            elif close[i] < lowest_low[i] and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.25
    
    return signals