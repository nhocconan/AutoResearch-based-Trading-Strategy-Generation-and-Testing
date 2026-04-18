#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_v1
Hypothesis: Use 4h Donchian breakout with volume confirmation and trend filter for directional entries. 
Long when price breaks above Donchian upper band (20), volume > 1.5x 20-period average, and price > 4h EMA34 (trend filter).
Short when price breaks below Donchian lower band (20), volume > 1.5x 20-period average, and price < 4h EMA34.
Exit on opposite Donchian breakout. Uses 1d ADX > 25 to ensure trending market regime.
Target: 20-40 trades/year by combining breakout with volume and trend filters to reduce false signals.
Works in bull via long breakouts and in bear via short breakouts during downtrends.
"""

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
    
    # Get 4h data for Donchian channels and EMA
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    upper_4h = np.full_like(high_4h, np.nan)
    lower_4h = np.full_like(low_4h, np.nan)
    
    if len(high_4h) >= donch_len:
        for i in range(donch_len, len(high_4h)):
            upper_4h[i] = np.max(high_4h[i-donch_len:i])
            lower_4h[i] = np.min(low_4h[i-donch_len:i])
    
    # 4h EMA34 for trend filter
    ema_len = 34
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_4h[ema_len-1] = np.mean(close_4h[:ema_len])
        for i in range(ema_len, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Align 4h indicators to 4h timeframe (same as primary)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) calculation
    adx_len = 14
    if len(high_1d) >= adx_len * 2:
        # True Range
        tr = np.maximum(high_1d[1:] - low_1d[1:], 
                        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                   np.abs(low_1d[1:] - close_1d[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= adx_len:
            # Initial averages
            atr[adx_len-1] = np.nanmean(tr[1:adx_len])
            dm_plus_smooth[adx_len-1] = np.nanmean(dm_plus[1:adx_len])
            dm_minus_smooth[adx_len-1] = np.nanmean(dm_minus[1:adx_len])
            
            # Wilder smoothing
            for i in range(adx_len, len(tr)):
                atr[i] = (atr[i-1] * (adx_len - 1) + tr[i]) / adx_len
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (adx_len - 1) + dm_plus[i]) / adx_len
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (adx_len - 1) + dm_minus[i]) / adx_len
        
        # DI and DX
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = (atr != 0) & ~np.isnan(atr)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx_valid = (di_plus + di_minus) != 0
        dx[dx_valid & ~np.isnan(di_plus) & ~np.isnan(di_minus)] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        # ADX: smoothed DX
        adx_1d = np.full_like(dx, np.nan)
        if len(dx) >= adx_len:
            valid_dx = ~np.isnan(dx)
            if np.sum(valid_dx) >= adx_len:
                adx_1d[adx_len-1] = np.nanmean(dx[valid_dx][:adx_len])
                for i in range(adx_len, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_1d[i] = (adx_1d[i-1] * (adx_len - 1) + dx[i]) / adx_len
    else:
        adx_1d = np.full_like(close_1d, np.nan)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, ema_len, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and trending:
            # Long: price breaks above Donchian upper + volume + price > EMA34 (uptrend)
            if close[i] > upper_4h_aligned[i] and vol_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume + price < EMA34 (downtrend)
            elif close[i] < lower_4h_aligned[i] and vol_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower
            if close[i] < lower_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper
            if close[i] > upper_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0