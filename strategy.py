#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian channel breakout with 12-hour trend filter and 1-day volume confirmation.
# Donchian breakouts capture momentum in trending markets, while the 12h ADX filter ensures
# we only trade in trending conditions (avoiding whipsaws in ranges). Volume confirmation
# ensures institutional participation. Designed for 6h timeframe to target 50-150 trades over 4 years.

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
    
    # 12-hour ADX(14) for trend strength filtering
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_12h), np.nan)
    dm_plus = np.full(len(close_12h), np.nan)
    dm_minus = np.full(len(close_12h), np.nan)
    
    if len(close_12h) > 1:
        tr[0] = high_12h[0] - low_12h[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_12h)):
            tr[i] = max(high_12h[i] - low_12h[i],
                       abs(high_12h[i] - close_12h[i-1]),
                       abs(low_12h[i] - close_12h[i-1]))
            dm_plus[i] = max(high_12h[i] - high_12h[i-1], 0)
            dm_minus[i] = max(low_12h[i-1] - low_12h[i], 0)
            dm_plus[i] = dm_plus[i] if dm_plus[i] > dm_minus[i] else 0
            dm_minus[i] = dm_minus[i] if dm_minus[i] > dm_plus[i] else 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_12h = np.full(len(close_12h), np.nan)
    s_dm_plus = np.full(len(close_12h), np.nan)
    s_dm_minus = np.full(len(close_12h), np.nan)
    
    if len(close_12h) >= 14:
        atr_12h[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_12h)):
            atr_12h[i] = atr_12h[i-1] - (atr_12h[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_12h), np.nan)
    di_minus = np.full(len(close_12h), np.nan)
    dx = np.full(len(close_12h), np.nan)
    
    for i in range(13, len(close_12h)):
        if atr_12h[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_12h[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_12h[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation (Wilder's smoothing)
    adx = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_12h)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channel (20-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):  # 20-period lookback
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19)  # ADX needs 27, Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian breakdown or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian breakout or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in trending markets with volume confirmation
            if trending_market and volume_filter:
                # Long: Donchian breakout
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Donchian breakdown
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals