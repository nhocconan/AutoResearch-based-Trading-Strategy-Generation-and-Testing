#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation.
Long when price breaks above 6h Donchian upper band AND 1w ADX > 25 AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian lower band AND 1w ADX > 25 AND volume > 1.5x 20-period average.
Exit when price retraces to 6h Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to target 12-37 trades/year on 6h timeframe.
Donchian channels provide clear breakout levels, weekly ADX filters for trending markets only,
volume confirmation avoids false breakouts. Works in both bull (breakouts) and bear (strong trends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1w ADX(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for TR smoothing + ADX
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - smoothed TR
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_smoothed = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop
    tr1_6h = np.abs(high - low)
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 30, 20, 14)  # Donchian needs 20, 1w ADX needs ~30, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr_6h[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        donchian_mid_val = donchian_mid[i]
        adx_val = adx_1w_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND strong trend (ADX > 25) AND volume spike (1.5x avg)
            if close[i] > donchian_upper and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND strong trend (ADX > 25) AND volume spike (1.5x avg)
            elif close[i] < donchian_lower and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= donchian_mid_val:
                exit_signal = True
            elif position == -1 and close[i] >= donchian_mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1wADX25_Trend_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0