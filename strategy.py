#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Long when price breaks above 4h 20-bar high AND 12h ADX > 25 AND volume > 1.5x average
# Short when price breaks below 4h 20-bar low AND 12h ADX > 25 AND volume > 1.5x average
# Exit when price crosses opposite Donchian level or ADX falls below 20
# Uses ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# Volume confirmation ensures breakouts have conviction
# Target: 100-200 total trades over 4 years (25-50/year) within optimal range

name = "4h_donchian20_12h_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    dm_plus_smooth[tr_period-1] = np.mean(dm_plus[:tr_period])
    dm_minus_smooth[tr_period-1] = np.mean(dm_minus[:tr_period])
    
    # Wilder smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # Calculate DI and DX
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    for i in range(tr_period, len(atr)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if ADX data not available
        if np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_confirm = volume[i] > 1.5 * volume_ma[i] if not np.isnan(volume_ma[i]) else False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below Donchian low OR ADX weakens (<20)
            if close[i] <= donchian_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian high OR ADX weakens (<20)
            if close[i] >= donchian_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume filters
            # Long: price breaks above Donchian high AND ADX > 25 AND volume confirmation
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 25 and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND ADX > 25 AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 25 and vol_confirm):
                signals[i] = -0.25
                position = -1
    
    return signals