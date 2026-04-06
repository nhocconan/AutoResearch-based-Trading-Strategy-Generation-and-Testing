#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R(14) mean reversion with 4h ADX(14) trend filter and 1d volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean-reversion entries.
# ADX filters for trending vs ranging markets to avoid false signals.
# Volume confirmation ensures institutional participation.
# Designed for 1h timeframe to target 60-150 trades over 4 years with low frequency.
# Uses 4h/1d for signal direction, 1h for entry timing.
# Includes session filter (08-20 UTC) to reduce noise trades.

name = "1h_willr4h_adx1d_vol_v1"
timeframe = "1h"
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
    
    # 4h Williams %R(14) for mean-reversion signals
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R calculation
    highest_high = np.full(len(close_4h), np.nan)
    lowest_low = np.full(len(close_4h), np.nan)
    willr_4h = np.full(len(close_4h), np.nan)
    
    for i in range(13, len(close_4h)):  # 14-period lookback
        highest_high[i] = np.max(high_4h[i-13:i+1])
        lowest_low[i] = np.min(low_4h[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            willr_4h[i] = -100 * (highest_high[i] - close_4h[i]) / (highest_high[i] - lowest_low[i])
    
    willr_4h_aligned = align_htf_to_ltf(prices, df_4h, willr_4h)
    
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
    adx_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 27:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[13:]  # Skip first 14 where DX is NaN
        if len(dx_valid) >= 14:
            adx_1d[26] = np.nanmean(dx_valid[:14])  # First ADX at index 26
            for i in range(27, len(close_1d)):
                adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(4, len(vol_1d)):  # 5-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-4:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 13, 4)  # ADX needs 27, Williams %R needs 13, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(willr_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume condition: current volume > 1.3x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.3
        
        # ADX filter: only trade when trending (ADX > 20) or extreme oversold/overbought
        trending_market = adx_1d_aligned[i] > 20
        extreme_oversold = willr_4h_aligned[i] < -80
        extreme_overbought = willr_4h_aligned[i] > -20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R overbought or stoploss
            if (willr_4h_aligned[i] > -20 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: Williams %R oversold or stoploss
            if (willr_4h_aligned[i] < -80 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in trending markets or extreme conditions
            if in_session and volume_filter:
                if trending_market or extreme_oversold or extreme_overbought:
                    # Long: oversold in uptrend or extreme oversold
                    if (willr_4h_aligned[i] < -80 and 
                        (trending_market and close[i] > close[i-1]) or extreme_oversold):
                        signals[i] = 0.20
                        position = 1
                        entry_price = close[i]
                    # Short: overbought in downtrend or extreme overbought
                    elif (willr_4h_aligned[i] > -20 and 
                          (trending_market and close[i] < close[i-1]) or extreme_overbought):
                        signals[i] = -0.20
                        position = -1
                        entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals