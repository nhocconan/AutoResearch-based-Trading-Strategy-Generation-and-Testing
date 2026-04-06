#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ATR breakout with 1-day volume confirmation and 1-week trend filter.
# Uses ATR-based breakouts for trend capture with volume confirmation to filter false breakouts.
# ADX on weekly timeframe ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Designed for low frequency (target: 50-150 trades over 4 years) with clear entry/exit rules.

name = "12h_atr_breakout_vol1d_trend1w_v1"
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
    
    # 1-day ATR(14) for breakout thresholds
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) > 1:
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i],
                          abs(high_1d[i] - close_1d[i-1]),
                          abs(low_1d[i] - close_1d[i-1]))
    
    # ATR calculation with Wilder's smoothing
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        atr_1d[13] = np.nanmean(tr_1d[1:15])  # First ATR at index 13
        for i in range(14, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1-day volume average for confirmation (20-period SMA)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1-week ADX(14) for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr_1w = np.full(len(close_1w), np.nan)
    dm_plus_1w = np.full(len(close_1w), np.nan)
    dm_minus_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) > 1:
        tr_1w[0] = high_1w[0] - low_1w[0]
        dm_plus_1w[0] = 0
        dm_minus_1w[0] = 0
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i],
                          abs(high_1w[i] - close_1w[i-1]),
                          abs(low_1w[i] - close_1w[i-1]))
            dm_plus_1w[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus_1w[i] = max(low_1w[i-1] - low_1w[i], 0)
    
    # Wilder's smoothing for TR, DM+, DM-
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus_1w = np.full(len(close_1w), np.nan)
    s_dm_minus_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        atr_1w[13] = np.nanmean(tr_1w[1:15])
        s_dm_plus_1w[13] = np.nanmean(dm_plus_1w[1:15])
        s_dm_minus_1w[13] = np.nanmean(dm_minus_1w[1:15])
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
            s_dm_plus_1w[i] = (s_dm_plus_1w[i-1] * 13 + dm_plus_1w[i]) / 14
            s_dm_minus_1w[i] = (s_dm_minus_1w[i-1] * 13 + dm_minus_1w[i]) / 14
    
    # DI+ and DI-
    di_plus_1w = np.full(len(close_1w), np.nan)
    di_minus_1w = np.full(len(close_1w), np.nan)
    dx_1w = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus_1w[i] = 100 * s_dm_plus_1w[i] / atr_1w[i]
            di_minus_1w[i] = 100 * s_dm_minus_1w[i] / atr_1w[i]
            if di_plus_1w[i] + di_minus_1w[i] != 0:
                dx_1w[i] = 100 * abs(di_plus_1w[i] - di_minus_1w[i]) / (di_plus_1w[i] + di_minus_1w[i])
    
    # ADX calculation
    adx_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx_1w[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx_1w[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_1w)):
                adx_1w[i] = (adx_1w[i-1] * 13 + dx_1w[i]) / 14
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19, 13)  # ADX needs 27, volume MA needs 19, ATR needs 13
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Trend filter: only trade when trending (ADX > 25)
        trending_market = adx_1w_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below open OR stoploss
            if (close[i] < prices['open'].iloc[i] or 
                close[i] < entry_price - 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above open OR stoploss
            if (close[i] > prices['open'].iloc[i] or 
                close[i] > entry_price + 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume in trending market
            if trending_market and volume_filter:
                # Calculate breakout levels based on ATR
                atr_value = atr_1d_aligned[i]
                
                # Long: price breaks above high + ATR
                if close[i] > high[i-1] + atr_value:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below low - ATR
                elif close[i] < low[i-1] - atr_value:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals