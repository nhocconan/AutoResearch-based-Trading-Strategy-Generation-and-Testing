#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 10-period RSI combined with 12-hour volume ratio and 4-hour ADX trend filter.
# RSI(10) identifies short-term momentum extremes for mean-reversion entries.
# ADX(14) filters for trending vs ranging markets to avoid false signals.
# Volume ratio > 1.5 confirms institutional participation.
# Designed for 4h timeframe to target 75-200 trades over 4 years with controlled frequency.
# Works in bull markets via trend-following and bear markets via mean-reversion at extremes.

name = "4h_rsi10_adx14_vol_ratio_v1"
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
    
    # 4-hour RSI(10) for momentum signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[9] = np.mean(gain[0:10])  # First average at index 9
    avg_loss[9] = np.mean(loss[0:10])
    
    for i in range(10, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 9 + gain[i]) / 10
        avg_loss[i] = (avg_loss[i-1] * 9 + loss[i]) / 10
    
    rs = np.full_like(close, np.nan)
    rsi = np.full_like(close, np.nan)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
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
    
    # Smoothed TR, DM+, DM-
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
    
    # ADX calculation
    adx_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx_12h[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_12h)):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 12-hour volume average for confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(4, len(vol_12h)):  # 5-period average
        vol_ma_12h[i] = np.mean(vol_12h[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 9, 4)  # ADX needs 27, RSI needs 9, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 12h average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: only trade when trending (ADX > 25) or extreme RSI
        trending_market = adx_aligned[i] > 25
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought or stoploss
            if (rsi[i] > 70 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI oversold or stoploss
            if (rsi[i] < 30 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in trending markets or extreme RSI
            if trending_market or oversold or overbought:
                if volume_filter:
                    # Long: oversold in uptrend or extreme oversold
                    if (rsi[i] < 30 and 
                        (trending_market and close[i] > close[i-1]) or oversold):
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: overbought in downtrend or extreme overbought
                    elif (rsi[i] > 70 and 
                          (trending_market and close[i] < close[i-1]) or overbought):
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals