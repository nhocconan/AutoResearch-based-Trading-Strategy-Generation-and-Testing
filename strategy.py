#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly volume confirmation and ADX trend filter.
# Uses 1-day timeframe targeting 30-100 total trades over 4 years.
# Long when price breaks above 20-day high with volume > 1.5x weekly average and ADX > 25.
# Short when price breaks below 20-day low with volume > 1.5x weekly average and ADX > 25.
# Trend filter prevents whipsaws in ranging markets.
# Includes ATR-based stoploss (2.5x ATR) to manage risk.
# Designed to work in both bull and bear markets by capturing strong trending moves.

name = "1d_donchian20_vol_adx_filter_v1"
timeframe = "1d"
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
    
    # 1-day Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-week ADX(14) for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr = np.full(len(close_1w), np.nan)
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
    
    # Directional Movement
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    if len(close_1w) > 1:
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1w)):
            dm_plus[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus[i] = max(low_1w[i-1] - low_1w[i], 0)
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        # Initial values (simple average of first 14 periods)
        atr_1w[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        
        # Wilder's smoothing: new_value = (prev_value * 13 + current_value) / 14
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
            s_dm_plus[i] = (s_dm_plus[i-1] * 13 + dm_plus[i]) / 14
            s_dm_minus[i] = (s_dm_minus[i-1] * 13 + dm_minus[i]) / 14
    
    # Directional Indicators
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1w[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1w[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation (smoothed DX)
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:  # Need 14 for DX + 14 for smoothing
        # First ADX is average of first 14 DX values
        valid_dx = dx[13:]  # Skip first 14 where DX is NaN
        if len(valid_dx) >= 14:
            adx[26] = np.nanmean(valid_dx[:14])  # First ADX at index 26
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period simple average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # ATR for stoploss (using daily data)
    tr_daily = np.full(n, np.nan)
    if n > 1:
        tr_daily[0] = high[0] - low[0]
        for i in range(1, n):
            tr_daily[i] = max(high[i] - low[i],
                             abs(high[i] - close[i-1]),
                             abs(low[i] - close[i-1]))
    
    atr_daily = np.full(n, np.nan)
    if n >= 14:
        atr_daily[13] = np.nansum(tr_daily[1:14])
        for i in range(14, n):
            atr_daily[i] = (atr_daily[i-1] * 13 + tr_daily[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 20 for Donchian, 27 for ADX)
    start = max(20, 27)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_daily[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when trending strongly (ADX > 25)
        trending_filter = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] < entry_price - 2.5 * atr_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] >= highest_high[i] or 
                close[i] > entry_price + 2.5 * atr_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend confirmation
            if volume_filter and trending_filter:
                # Long: price breaks above 20-day high
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below 20-day low
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals