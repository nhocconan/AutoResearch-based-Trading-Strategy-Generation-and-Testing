#!/usr/bin/env python3
"""
4h_1d_keltner_squeeze_breakout_v1
Strategy: 4h Keltner channel squeeze breakout with volume confirmation and 1d ADX trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines volatility contraction (Keltner squeeze) with breakout logic. Uses 1d ADX to filter for trending markets only, avoiding false breakouts in ranging conditions. Volume confirmation ensures breakout validity. Designed to work in both bull and bear markets by capturing explosive moves after low volatility periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Keltner Channel (20, 2.0) on 4h
    atr_period = 20
    atr = pd.Series(high - low).rolling(window=atr_period, min_periods=atr_period).mean().values
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_keltner = ma + 2.0 * atr
    lower_keltner = ma - 2.0 * atr
    
    # Bollinger Bands (20, 2.0) on 4h for squeeze detection
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma + 2.0 * bb_std
    lower_bb = ma - 2.0 * bb_std
    
    # Squeeze condition: Bollinger Bands inside Keltner Channels
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d ADX for trend filtering
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
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Squeeze breakout conditions
        squeeze_release = not squeeze[i-1] and squeeze[i]  # Squeeze just released
        breakout_up = price_close > upper_keltner[i]
        breakout_down = price_close < lower_keltner[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout after squeeze release in trending market
        long_signal = squeeze_release and breakout_up and vol_confirmed and trending
        
        # Short: downward breakout after squeeze release in trending market
        short_signal = squeeze_release and breakout_down and vol_confirmed and trending
        
        # Exit when price returns to the middle of Keltner channel
        exit_long = position == 1 and price_close < ma[i]
        exit_short = position == -1 and price_close > ma[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals