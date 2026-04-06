#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price position within daily Donchian channel combined with 1-week ADX trend filter and volume confirmation.
# In trending markets (ADX > 25), buy near lower band (oversold pullback) and sell near upper band (overbought bounce).
# In ranging markets (ADX < 20), fade extremes at Donchian boundaries for mean reversion.
# Volume confirmation filters low-liquidity false breakouts.
# Designed for 12h timeframe to target 50-150 trades over 4 years with controlled frequency.

name = "12h_donchian_pos_adx1w_vol_v1"
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
    
    # 1-day Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high and low
    highest_high = np.full(len(close_1d), np.nan)
    lowest_low = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):  # 20-period lookback
        highest_high[i] = np.max(high_1d[i-19:i+1])
        lowest_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate price position within channel (0 = at low, 1 = at high)
    donch_pos = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            donch_pos[i] = (close_1d[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            donch_pos[i] = 0.5  # Avoid division by zero
    
    donch_pos_aligned = align_htf_to_ltf(prices, df_1d, donch_pos)
    
    # 1-week ADX(14) for trend strength
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
    
    # Smoothed TR, DM+, DM-
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        atr_1w[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = atr_1w[i-1] - (atr_1w[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
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
    
    # ADX calculation
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:
        dx_valid = dx[13:]
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19, 4)  # ADX needs 27, Donchian needs 19, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_pos_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.2
        
        # Market regime
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price at upper Donchian band or stoploss
            if (donch_pos_aligned[i] > 0.8 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price at lower Donchian band or stoploss
            if (donch_pos_aligned[i] < 0.2 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if volume_filter:
                if trending:
                    # In uptrend: buy dips near lower band
                    if donch_pos_aligned[i] < 0.3 and close[i] > close[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # In downtrend: sell rallies near upper band
                    elif donch_pos_aligned[i] > 0.7 and close[i] < close[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                elif ranging:
                    # In range: fade extremes
                    if donch_pos_aligned[i] < 0.2:
                        signals[i] = 0.25  # Buy at support
                        position = 1
                        entry_price = close[i]
                    elif donch_pos_aligned[i] > 0.8:
                        signals[i] = -0.25  # Sell at resistance
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals