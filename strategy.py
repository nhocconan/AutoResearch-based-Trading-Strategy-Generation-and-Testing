#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian(20) breakout from 1d + 1d ADX trend filter + volume confirmation.
# Long when price breaks above 1d Donchian upper (20-period high) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average volume.
# Short when price breaks below 1d Donchian lower (20-period low) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average volume.
# Exit when price crosses the 1d Donchian midpoint (median of upper/lower) OR ADX < 20 (trend weakening).
# Uses discrete position size 0.25. Donchian provides clear breakout levels, ADX filters for trending markets, volume confirms conviction.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (capture breakouts) and bear markets (capture breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels, ADX, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Upper = 20-period high, Lower = 20-period low
    lookback = 20
    upper_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    midpoint_1d = (upper_1d + lower_1d) / 2.0
    
    # === 1d Indicators: ADX(14) for trend strength ===
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    # True Range
    tr1 = pd.Series(high_1d).diff(1).abs()
    tr2 = pd.Series(low_1d).diff(1).abs()
    tr3 = pd.Series(close_1d).diff(1).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff(1)
    down_move = pd.Series(low_1d).diff(1).abs()  # negative of low diff
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_period = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_period)
    minus_di = 100 * (minus_dm_smooth / tr_period)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # === 1d Indicators: Volume average (20-period) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian(20) + ADX(14) need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        midpoint = midpoint_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > (1.5 * vol_ma) if not np.isnan(vol_ma) else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < midpoint (breakdown) OR ADX < 20 (trend weakening)
            if (price < midpoint) or (adx_val < 20):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > midpoint (breakout) OR ADX < 20 (trend weakening)
            if (price > midpoint) or (adx_val < 20):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > upper (breakout) AND ADX > 25 (trending) AND volume confirmed
            if (price > upper) and (adx_val > 25) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < lower (breakdown) AND ADX > 25 (trending) AND volume confirmed
            elif (price < lower) and (adx_val > 25) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_ADX25_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0