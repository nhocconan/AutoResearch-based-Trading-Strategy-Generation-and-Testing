#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w ADX filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 1w ADX > 25 AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND 1w ADX > 25 AND volume > 1.5x average
# Exit when price crosses opposite Donchian band OR ADX < 20 (trend weakening)
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in trending markets (ADX filter) with volume confirmation to avoid false breakouts

name = "12h_donchian_adx_vol_v2"
timeframe = "12h"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX (14-period) from 1w timeframe for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = weekly_high[0] - weekly_low[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low),
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)),
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + strong trend (ADX > 25) + volume confirmation
            # Long: break above upper band
            if (close[i] > highest_high[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band
            elif (close[i] < lowest_low[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals