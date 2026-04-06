#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day ADX trend filter and volume confirmation
# Donchian(20) breakouts capture momentum in both bull and bear markets
# ADX(14) from daily timeframe filters for trending conditions (ADX > 25)
# Volume confirmation requires current volume > 1.5x 20-period average
# Designed for 4h timeframe to target 75-200 trades over 4 years with controlled frequency

name = "4h_donchian20_1d_adx_vol_v1"
timeframe = "4h"
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
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Donchian breakout signals
    long_breakout = close > highest_high  # Close above previous 20-period high
    short_breakout = close < lowest_low   # Close below previous 20-period low
    
    # 1-day ADX(14) for trend strength filtering
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1d), np.nan)
    dm_plus = np.full(len(close_1d), np.nan)
    dm_minus = np.full(len(close_1d), np.nan)
    
    if len(close_1d) > 1:
        tr[0] = high_1d[0] - low_1d[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i],
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
            dm_plus[i] = max(high_1d[i] - high_1d[i-1], 0)
            dm_minus[i] = max(low_1d[i-1] - low_1d[i], 0)
            # + and - DM cannot both be positive
            if dm_plus[i] > 0 and dm_minus[i] > 0:
                if dm_plus[i] > dm_minus[i]:
                    dm_minus[i] = 0
                else:
                    dm_plus[i] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_1d = np.full(len(close_1d), np.nan)
    s_dm_plus = np.full(len(close_1d), np.nan)
    s_dm_minus = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        # Initial values (simple average of first 14 periods)
        atr_1d[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        # Wilder's smoothing for subsequent periods
        for i in range(14, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
            s_dm_plus[i] = (s_dm_plus[i-1] * 13 + dm_plus[i]) / 14
            s_dm_minus[i] = (s_dm_minus[i-1] * 13 + dm_minus[i]) / 14
    
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
        # First ADX is average of first 14 DX values
        adx[26] = np.nanmean(dx[13:27])
        # Wilder's smoothing for subsequent ADX values
        for i in range(27, len(close_1d)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_filter = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 27 for ADX + 19 for Donchian)
    start = max(27, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Donchian breakdown or stoploss
            if (short_breakout[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Donchian breakout or stoploss
            if (long_breakout[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in trending markets with volume confirmation
            if trending_market and volume_filter[i]:
                # Long: Donchian breakout
                if long_breakout[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Donchian breakdown
                elif short_breakout[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals