#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian channel breakout with 12-hour ADX trend filter and 1-day volume confirmation.
# Donchian breakouts capture momentum in trending markets.
# ADX ensures we only trade in trending conditions (ADX > 25) to avoid whipsaws.
# Volume confirmation ensures institutional participation in breakouts.
# Designed for 6h timeframe to target 50-150 trades over 4 years with medium frequency.

name = "6h_donchian20_12h_adx1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 12-hour ADX(14) for trend strength filtering
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and Directional Movement
    tr_12h = np.full(len(close_12h), np.nan)
    dm_plus_12h = np.full(len(close_12h), np.nan)
    dm_minus_12h = np.full(len(close_12h), np.nan)
    
    if len(close_12h) > 1:
        tr_12h[0] = high_12h[0] - low_12h[0]
        dm_plus_12h[0] = 0
        dm_minus_12h[0] = 0
        for i in range(1, len(close_12h)):
            tr_12h[i] = max(high_12h[i] - low_12h[i],
                           abs(high_12h[i] - close_12h[i-1]),
                           abs(low_12h[i] - close_12h[i-1]))
            dm_plus_12h[i] = max(high_12h[i] - high_12h[i-1], 0)
            dm_minus_12h[i] = max(low_12h[i-1] - low_12h[i], 0)
            if dm_plus_12h[i] > dm_minus_12h[i]:
                dm_minus_12h[i] = 0
            else:
                dm_plus_12h[i] = 0
    
    # Smoothed TR, DM+, DM-
    atr_12h = np.full(len(close_12h), np.nan)
    s_dm_plus_12h = np.full(len(close_12h), np.nan)
    s_dm_minus_12h = np.full(len(close_12h), np.nan)
    
    if len(close_12h) >= 14:
        atr_12h[13] = np.nansum(tr_12h[1:14])
        s_dm_plus_12h[13] = np.nansum(dm_plus_12h[1:14])
        s_dm_minus_12h[13] = np.nansum(dm_minus_12h[1:14])
        for i in range(14, len(close_12h)):
            atr_12h[i] = atr_12h[i-1] - (atr_12h[i-1]/14) + tr_12h[i]
            s_dm_plus_12h[i] = s_dm_plus_12h[i-1] - (s_dm_plus_12h[i-1]/14) + dm_plus_12h[i]
            s_dm_minus_12h[i] = s_dm_minus_12h[i-1] - (s_dm_minus_12h[i-1]/14) + dm_minus_12h[i]
    
    # DI+ and DI-
    di_plus_12h = np.full(len(close_12h), np.nan)
    di_minus_12h = np.full(len(close_12h), np.nan)
    dx_12h = np.full(len(close_12h), np.nan)
    
    for i in range(14, len(close_12h)):
        if atr_12h[i] != 0:
            di_plus_12h[i] = 100 * s_dm_plus_12h[i] / atr_12h[i]
            di_minus_12h[i] = 100 * s_dm_minus_12h[i] / atr_12h[i]
            if di_plus_12h[i] + di_minus_12h[i] != 0:
                dx_12h[i] = 100 * abs(di_plus_12h[i] - di_minus_12h[i]) / (di_plus_12h[i] + di_minus_12h[i])
    
    # ADX calculation
    adx_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 28:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx_12h[14:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx_12h[27] = np.nanmean(dx_valid[:14])  # First ADX at index 27
            for i in range(28, len(close_12h)):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx_12h[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(28, 20, 19)  # ADX needs 28, Donchian needs 20, volume needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when strongly trending (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian breakdown or stoploss
            if (close[i] < donchian_low[i-1] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian breakout or stoploss
            if (close[i] > donchian_high[i-1] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend
            if volume_filter and strong_trend:
                # Long: breakout above Donchian high
                if close[i] > donchian_high[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low
                elif close[i] < donchian_low[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals