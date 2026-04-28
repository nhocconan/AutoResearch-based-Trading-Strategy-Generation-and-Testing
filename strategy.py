#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian channel breakout with 1w ADX trend filter and volume confirmation.
# Enter long when price breaks above 1d Donchian(20) upper band with volume spike and 1w ADX > 25.
# Enter short when price breaks below 1d Donchian(20) lower band with volume spike and 1w ADX > 25.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
# ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging conditions.

name = "6h_Donchian20_1wADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    n_1d = len(high_1d)
    upper_band = np.full(n_1d, np.nan)
    lower_band = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        start_idx = max(0, i - 19)  # 20-period lookback
        upper_band[i] = np.max(high_1d[start_idx:i+1])
        lower_band[i] = np.min(low_1d[start_idx:i+1])
    
    # Forward fill to get most recent Donchian levels
    upper_band = pd.Series(upper_band).ffill().values
    lower_band = pd.Series(lower_band).ffill().values
    
    # Align 1d Donchian bands to 6h timeframe with 1-bar delay for confirmation
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    if n_1w < 14:
        return np.zeros(n)
    
    # Calculate True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    atr = np.full(n_1w, np.nan)
    dm_plus_smooth = np.full(n_1w, np.nan)
    dm_minus_smooth = np.full(n_1w, np.nan)
    
    # Initial values (simple average of first 14 periods)
    if n_1w >= 14:
        atr[13] = np.nanmean(tr[1:15])
        dm_plus_smooth[13] = np.nanmean(dm_plus[1:15])
        dm_minus_smooth[13] = np.nanmean(dm_minus[1:15])
        
        # Wilder's smoothing for subsequent values
        for i in range(14, n_1w):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.full(n_1w, np.nan)
    di_minus = np.full(n_1w, np.nan)
    
    for i in range(14, n_1w):
        if atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
    
    # Calculate DX and ADX
    dx = np.full(n_1w, np.nan)
    for i in range(14, n_1w):
        if (di_plus[i] + di_minus[i]) != 0:
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    adx = np.full(n_1w, np.nan)
    # Initial ADX (average of first 14 DX values after period 14)
    if n_1w >= 28:
        adx[27] = np.nanmean(dx[14:28])
        # Wilder's smoothing for subsequent ADX values
        for i in range(28, n_1w):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > upper_aligned[i] and volume_spike[i]
        short_breakout = close[i] < lower_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian band or loss of trend
        long_exit = close[i] < lower_aligned[i] or not trending
        short_exit = close[i] > upper_aligned[i] or not trending
        
        # Handle entries and exits
        if long_breakout and trending and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and trending and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals