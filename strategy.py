#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-week ADX trend filter and 1-day volume confirmation.
# In bull markets: buy breakouts above 20-period high with ADX > 25 (strong trend).
# In bear markets: sell breakdowns below 20-period low with ADX > 25.
# Volume confirmation ensures institutional participation. Designed for low frequency (10-30 trades/year).

name = "12h_donchian20_1w_adx1d_vol_v1"
timeframe = "12h"
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
    
    # 1-week Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels
    highest_high = np.full(len(high_1w), np.nan)
    lowest_low = np.full(len(low_1w), np.nan)
    
    for i in range(19, len(high_1w)):  # 20-period lookback
        highest_high[i] = np.max(high_1w[i-19:i+1])
        lowest_low[i] = np.min(low_1w[i-19:i+1])
    
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # 1-day ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.full(len(close_1d), np.nan)
    if len(close_1d) > 1:
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i],
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    dm_plus = np.full(len(close_1d), np.nan)
    dm_minus = np.full(len(close_1d), np.nan)
    if len(close_1d) > 1:
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1d)):
            dm_plus[i] = max(high_1d[i] - high_1d[i-1], 0)
            dm_minus[i] = max(low_1d[i-1] - low_1d[i], 0)
    
    # Smoothed TR, DM+ (Wilder's smoothing)
    atr_1d = np.full(len(close_1d), np.nan)
    s_dm_plus = np.full(len(close_1d), np.nan)
    s_dm_minus = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        atr_1d[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1d)):
            atr_1d[i] = atr_1d[i-1] - (atr_1d[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1d), np.nan)
    di_minus = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    for i in range(13, len(close_1d)):
        if atr_1d[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1d[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1d[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation (smoothed DX)
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_1d)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-day volume average for confirmation (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of all indicators)
    start = max(27, 19)  # ADX needs 27, Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when trending strongly (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: breakdown below Donchian low or stoploss
            if (close[i] < lowest_low_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above Donchian high or stoploss
            if (close[i] > highest_high_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakouts with volume and trend
            if volume_filter and strong_trend:
                # Long: breakout above Donchian high
                if close[i] > highest_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below Donchian low
                elif close[i] < lowest_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals