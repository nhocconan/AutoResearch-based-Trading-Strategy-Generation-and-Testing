#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX(14) trend strength and Donchian channel calculation.
- Donchian: Upper = 20-period high, Lower = 20-period low on 1d data.
- ADX: Measures trend strength (>25 = strong trend) to filter breakouts.
- Entry: Long when price > 1d Upper Band AND ADX > 25 AND volume > 1.5 * 20-period average volume.
         Short when price < 1d Lower Band AND ADX > 25 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout (price < Upper Band for long exit, price > Lower Band for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (strong uptrend breakouts) and bear markets (strong downtrend breakouts) with ADX filter avoiding false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def true_range(high, low, close):
    """Calculate True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First period
    return tr

def adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # Calculate True Range
    tr = true_range(high, low, close)
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # First values
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    # Initial average
    atr[period] = np.mean(tr[1:period+1])
    plus_di[period] = np.mean(plus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
    minus_di[period] = np.mean(minus_dm[1:period+1]) / atr[period] * 100 if atr[period] != 0 else 0
    
    # Wilder's smoothing
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(tr)
    adx_val = np.zeros_like(tr)
    
    for i in range(period, len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx[i] = 0.0
    
    # ADX is smoothed DX
    adx_start = 2 * period - 1
    if adx_start < len(dx):
        adx_val[adx_start] = np.mean(dx[period:adx_start+1])
        for i in range(adx_start + 1, len(dx)):
            adx_val[i] = (adx_val[i-1] * (period - 1) + dx[i]) / period
    
    return adx_val

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need sufficient data for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    # 1d Donchian Channel (20-period)
    donchian_period = 20
    if len(df_1d) < donchian_period:
        return np.zeros(n)
    
    upper_band = pd.Series(df_1d['high'].values).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(df_1d['low'].values).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 1d ADX(14) for trend filter
    adx_period = 14
    if len(df_1d) < adx_period + 1:
        return np.zeros(n)
    
    adx_values = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, adx_period)
    
    # Align 1d indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, adx_period + 1, 20)  # Need all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price < Upper Band (breakdown)
            if position == 1:
                if curr_close < upper_band_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Lower Band (breakout above)
            elif position == -1:
                if curr_close > lower_band_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with ADX filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # ADX filter: trend strength > 25
            adx_filter = adx_aligned[i] > 25
            
            # Long: price > Upper Band AND ADX > 25 AND volume confirmation
            long_condition = (curr_close > upper_band_aligned[i] and 
                            adx_filter and
                            volume_confirm)
            
            # Short: price < Lower Band AND ADX > 25 AND volume confirmation
            short_condition = (curr_close < lower_band_aligned[i] and 
                             adx_filter and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dADX_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0