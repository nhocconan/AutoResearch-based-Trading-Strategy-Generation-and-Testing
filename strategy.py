#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R(14) mean-reversion signals filtered by 1-day ADX(14) trend strength.
# In trending markets (ADX > 25), we trade pullbacks: long when Williams %R < -80 (oversold),
# short when Williams %R > -20 (overbought). In ranging markets (ADX <= 25), we fade extremes:
# long at Williams %R < -90, short at Williams %R > -10. Volume confirmation (>1.5x 1-day average)
# ensures institutional participation. Designed for low frequency (~20-40 trades/year) with
# clear entry/exit rules to minimize fee drag. Williams %R identifies mean-reversion
# opportunities, ADX filters for appropriate market regime, and volume avoids false breakouts.

name = "12h_willr1d_adx1d_vol_v1"
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
    
    # 1-day Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation
    highest_high = np.full(len(close_1d), np.nan)
    lowest_low = np.full(len(close_1d), np.nan)
    willr = np.full(len(close_1d), np.nan)
    
    for i in range(13, len(close_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            willr[i] = -100 * (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i])
    
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    
    # 1-day ADX(14) for trend strength
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1d_adx), np.nan)
    dm_plus = np.full(len(close_1d_adx), np.nan)
    dm_minus = np.full(len(close_1d_adx), np.nan)
    
    if len(close_1d_adx) > 1:
        tr[0] = high_1d_adx[0] - low_1d_adx[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1d_adx)):
            tr[i] = max(high_1d_adx[i] - low_1d_adx[i],
                       abs(high_1d_adx[i] - close_1d_adx[i-1]),
                       abs(low_1d_adx[i] - close_1d_adx[i-1]))
            dm_plus[i] = max(high_1d_adx[i] - high_1d_adx[i-1], 0)
            dm_minus[i] = max(low_1d_adx[i-1] - low_1d_adx[i], 0)
            if dm_plus[i] > dm_minus[i]:
                dm_minus[i] = 0
            else:
                dm_plus[i] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing
    atr_1d = np.full(len(close_1d_adx), np.nan)
    s_dm_plus = np.full(len(close_1d_adx), np.nan)
    s_dm_minus = np.full(len(close_1d_adx), np.nan)
    
    if len(close_1d_adx) >= 14:
        atr_1d[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1d_adx)):
            atr_1d[i] = atr_1d[i-1] - (atr_1d[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1d_adx), np.nan)
    di_minus = np.full(len(close_1d_adx), np.nan)
    dx = np.full(len(close_1d_adx), np.nan)
    
    for i in range(13, len(close_1d_adx)):
        if atr_1d[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1d[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1d[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation (smoothed DX)
    adx = np.full(len(close_1d_adx), np.nan)
    if len(close_1d_adx) >= 27:
        dx_valid = dx[13:]
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])
            for i in range(27, len(close_1d_adx)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])  # 20-period average
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 13, 19)  # ADX needs 27, Williams %R needs 13, volume needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(willr_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: trending (ADX > 25) or ranging (ADX <= 25)
        trending_market = adx_aligned[i] > 25
        ranging_market = adx_aligned[i] <= 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R overbought or stoploss
            if (willr_aligned[i] > -20 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R oversold or stoploss
            if (willr_aligned[i] < -80 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on market regime
            if trending_market:
                # In trending markets, trade pullbacks
                if volume_filter:
                    # Long: pullback to oversold in uptrend
                    if (willr_aligned[i] < -80 and close[i] > close[i-1]):
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: pullback to overbought in downtrend
                    elif (willr_aligned[i] > -20 and close[i] < close[i-1]):
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:  # ranging market
                # In ranging markets, fade extremes
                if volume_filter:
                    # Long: extreme oversold
                    if willr_aligned[i] < -90:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: extreme overbought
                    elif willr_aligned[i] > -10:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals