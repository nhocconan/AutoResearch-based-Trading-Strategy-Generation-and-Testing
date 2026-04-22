#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w ADX filter and volume confirmation.
Long when price breaks above Donchian upper with strong ADX trend and volume spike.
Short when price breaks below Donchian lower with strong ADX trend and volume spike.
Exit when price crosses Donchian middle or ADX weakens.
Uses 1w ADX for trend strength filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (7-25/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for ADX filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on 1d
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1w ADX (14-period)
    high_w = pd.Series(df_weekly['high'].values)
    low_w = pd.Series(df_weekly['low'].values)
    close_w = pd.Series(df_weekly['close'].values)
    
    # True Range
    tr1 = high_w - low_w
    tr2 = abs(high_w - close_w.shift(1))
    tr3 = abs(low_w - close_w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_w = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_w.diff()
    down_move = -low_w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_w)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_w)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_w = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx_w.values)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with strong ADX and volume
            if (close[i] > dc_upper[i] and 
                adx_aligned[i] > 25 and  # Strong trend
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with strong ADX and volume
            elif (close[i] < dc_lower[i] and 
                  adx_aligned[i] > 25 and  # Strong trend
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle OR ADX weakens
                if close[i] < dc_middle[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle OR ADX weakens
                if close[i] > dc_middle[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_DonchianBreakout_1wADX_Volume"
timeframe = "1d"
leverage = 1.0
#%%