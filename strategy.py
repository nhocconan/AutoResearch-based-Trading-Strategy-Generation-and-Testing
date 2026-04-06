#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Bollinger Band squeeze breakout with 1-day volume confirmation and 1-week ADX trend filter.
# Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets.
# Volume confirmation ensures institutional participation in the breakout.
# ADX filter avoids false breakouts in ranging markets.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.

name = "12h_bb_squeeze_1d_vol_1w_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate basis (SMA)
    basis = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        basis[i] = np.mean(cl[i-bb_length+1:i+1])
    
    # Calculate standard deviation
    dev = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        dev[i] = np.std(cl[i-bb_length+1:i+1])
    
    # Calculate upper and lower bands
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (upper - lower) / basis
    bb_width_ma = np.full(n, np.nan)  # 50-period MA of BB width
    bb_length_ma = 50
    for i in range(bb_length_ma - 1, n):
        bb_width_ma[i] = np.mean(bb_width[i-bb_length_ma+1:i+1])
    
    # Squeeze condition: BB width below its 50-period MA
    squeeze = bb_width < bb_width_ma
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    vol_length_1d = 20
    for i in range(vol_length_1d - 1, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-vol_length_1d+1:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
            if dm_plus[i] > dm_minus[i]:
                dm_minus[i] = 0
            else:
                dm_plus[i] = 0
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_length_ma - 1, vol_length_1d - 1, 27)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper band or stoploss
            if (close[i] >= upper[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches lower band or stoploss
            if (close[i] <= lower[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries after squeeze
            if squeeze[i-1] and not squeeze[i]:  # Squeeze just released
                if volume_filter and trending_market:
                    # Long: breakout above upper band
                    if close[i] > upper[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: breakout below lower band
                    elif close[i] < lower[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals