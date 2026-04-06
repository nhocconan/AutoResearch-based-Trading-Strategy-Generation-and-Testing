#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ADX trend filter and 1-week volume confirmation.
# Donchian breakouts capture momentum in trending markets. ADX ensures we only trade when trend is strong.
# Volume confirmation filters false breakouts. Designed for 4h timeframe targeting 75-200 trades over 4 years.

name = "4h_donchian20_1d_adx1w_vol_v1"
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
            dm_plus[i] = dm_plus[i] if dm_plus[i] > dm_minus[i] else 0
            dm_minus[i] = dm_minus[i] if dm_minus[i] > dm_plus[i] else 0
    
    # Smoothed TR, DM+, DM-
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
    
    # ADX calculation
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_1d)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 4-hour Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19)  # ADX needs 27, Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 1.2
        
        # ADX filter: only trade when trending strongly (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < donch_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > donch_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend
            if volume_filter and strong_trend:
                # Long: price breaks above Donchian high
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low
                elif close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals