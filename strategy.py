#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour timeframe with 1-day pivot point (PP) and Fibonacci extension levels for trend continuation.
# Uses daily pivot points calculated from previous day's OHLC to identify key support/resistance levels.
# Enters long when price breaks above R1 with volume confirmation, short when breaks below S1.
# Uses 1-week ADX to filter for trending markets only, avoiding false breakouts in ranging conditions.
# Designed for 6h timeframe to target 50-150 trades over 4 years with proper risk management.

name = "6h_pivot_fib_ext_1d_adx1w_v1"
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
    
    # 1-day data for pivot points (calculated from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H + L + C)/3
    # Support/resistance levels: R1 = 2*PP - L, S1 = 2*PP - H
    # Fibonacci extensions: R2 = PP + (H - L), S2 = PP - (H - L)
    pp = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day's data
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = 2 * pp[i] - low_1d[i-1]
        s1[i] = 2 * pp[i] - high_1d[i-1]
        r2[i] = pp[i] + (high_1d[i-1] - low_1d[i-1])
        s2[i] = pp[i] - (high_1d[i-1] - low_1d[i-1])
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for non-look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1-week ADX(14) for trend strength filtering
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1w), np.nan)
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
            dm_plus[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus[i] = max(low_1w[i-1] - low_1w[i], 0)
            # Wilder smoothing: if +DM > -DM then -DM=0, else +DM=0
            if dm_plus[i] > dm_minus[i]:
                dm_minus[i] = 0
            else:
                dm_plus[i] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (14-period)
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        # Initial values: simple average of first 14 periods
        atr_1w[13] = np.nansum(tr[1:14])  # tr[1] to tr[14]
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        
        # Wilder smoothing: subsequent values
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
            s_dm_plus[i] = (s_dm_plus[i-1] * 13 + dm_plus[i]) / 14
            s_dm_minus[i] = (s_dm_minus[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1w[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1w[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation: smoothed DX
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:  # Need 14 for DX + 14 for smoothing
        # First ADX is average of first 14 DX values
        valid_dx_start = 13  # First valid DX at index 13
        if len(close_1w) >= valid_dx_start + 14:
            adx[26] = np.nanmean(dx[valid_dx_start:valid_dx_start+14])  # First ADX at index 26
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period: need 1 day for pivots, 27 for ADX
    start = max(27, 1)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 6-period average for breakout confirmation
        if i >= 6:
            vol_ma = np.mean(volume[i-6:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below S1 or stoploss
            if (close[i] < s1_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above R1 or stoploss
            if (close[i] > r1_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries in trending markets with volume confirmation
            if trending_market and volume_filter:
                # Long: price breaks above R1
                if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below S1
                elif close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals