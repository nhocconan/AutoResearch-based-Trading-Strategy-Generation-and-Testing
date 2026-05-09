#!/usr/bin/env python3
# Hypothesis: 12-hour Donchian channel breakout with 1-day ADX trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band (20-period) with daily ADX > 25 and volume > 1.5x average
# Short when price breaks below 12h Donchian lower band (20-period) with daily ADX > 25 and volume > 1.5x average
# Exit when price crosses the 12h Donchian midpoint or reverses to opposite band
# Uses Donchian channels for breakout signals, ADX for trend strength filtering, volume for conviction
# Designed to capture strong momentum moves in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Donchian_Breakout_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR and DM
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Wilder's smoothing
    for i in range(len(tr)):
        if i < tr_period:
            continue
        if i == tr_period:
            atr[i] = np.nansum(tr[i-tr_period+1:i+1])
            dm_plus_smooth[i] = np.nansum(dm_plus[i-tr_period+1:i+1])
            dm_minus_smooth[i] = np.nansum(dm_minus[i-tr_period+1:i+1])
        else:
            atr[i] = atr[i-1] - (atr[i-1]/tr_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1]/tr_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1]/tr_period) + dm_minus[i]
    
    # Calculate DI and DX
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(tr_period, len(tr)):
        if atr[i] > 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) > 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(dx, np.nan)
    for i in range(2*tr_period-1, len(dx)):
        if i == 2*tr_period-1:
            valid_dx = dx[tr_period:i+1]
            adx[i] = np.nanmean(valid_dx[~np.isnan(valid_dx)])
        else:
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
            else:
                adx[i] = adx[i-1]
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 2*14)  # Need enough data for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ADX > 25, volume spike
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ADX > 25, volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses Donchian mid or reverses to lower band
            if (close[i] <= donchian_mid[i]) or (close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses Donchian mid or reverses to upper band
            if (close[i] >= donchian_mid[i]) or (close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals