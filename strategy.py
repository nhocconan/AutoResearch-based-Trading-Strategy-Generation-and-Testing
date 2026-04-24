#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ADX(14) for trend filter (ADX > 25 = trending market).
- Entry: Long when price breaks above Donchian upper AND ADX > 25 AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower AND ADX > 25 AND volume > 1.5 * 4h volume MA(20).
- Exit: Opposite Donchian breakout (Long exits when price < Donchian lower, Short exits when price > Donchian upper).
- Signal size: 0.25 discrete to balance capture and fee control.
- Uses Donchian channels for structure, ADX to avoid ranging markets, volume to confirm conviction.
- Works in bull (buying strong breakouts) and bear (selling strong breakdowns) with reduced whipsaws from ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial sum
    tr_sum[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_sum[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_sum[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = np.zeros_like(dx)
    
    # Initial ADX
    adx[2*tr_period] = np.nanmean(dx[tr_period+1:2*tr_period+1])
    
    # Wilder smoothing for ADX
    for i in range(2*tr_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 4h
    donchian_period = 20
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume MA(20) on 4h
    vol_ma = np.zeros(n)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period - 1, 2*14 + 1, 19)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: ADX > 25 = trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if trending and vol_confirm:
                # Long: price breaks above Donchian upper
                if curr_high > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower
                elif curr_low < lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian lower
            if curr_low < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian upper
            if curr_high > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX25_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0