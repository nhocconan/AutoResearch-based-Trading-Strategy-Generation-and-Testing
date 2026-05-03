#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX regime filter.
# Long when price breaks above Donchian upper band with volume spike and ADX>25 (trending).
# Short when price breaks below Donchian lower band with volume spike and ADX>25.
# Uses 1d ADX for regime filter to avoid whipsaws in ranging markets.
# Designed for 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL.

name = "4h_Donchian20_VolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_smoothed = np.full_like(tr, np.nan)
    plus_dm_smoothed = np.full_like(plus_dm, np.nan)
    minus_dm_smoothed = np.full_like(minus_dm, np.nan)
    
    # Initialize first values
    tr_smoothed[period] = np.nansum(tr[1:period+1])
    plus_dm_smoothed[period] = np.nansum(plus_dm[1:period+1])
    minus_dm_smoothed[period] = np.nansum(minus_dm[1:period+1])
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        tr_smoothed[i] = tr_smoothed[i-1] - (tr_smoothed[i-1] / period) + tr[i]
        plus_dm_smoothed[i] = plus_dm_smoothed[i-1] - (plus_dm_smoothed[i-1] / period) + plus_dm[i]
        minus_dm_smoothed[i] = minus_dm_smoothed[i-1] - (minus_dm_smoothed[i-1] / period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan)
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d ADX to 4h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian(20) channels on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(upper) or np.isnan(lower) or np.isnan(adx_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_val > 25
        
        # Entry conditions
        if position == 0 and is_trending and vol_spike:
            # Long: breakout above upper band
            if close_val > upper:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band
            elif close_val < lower:
                signals[i] = -0.25
                position = -1
                
        # Exit conditions
        elif position == 1:
            # Long exit: breakdown below lower band or loss of trend
            if close_val < lower or adx_val <= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above upper band or loss of trend
            if close_val > upper or adx_val <= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals