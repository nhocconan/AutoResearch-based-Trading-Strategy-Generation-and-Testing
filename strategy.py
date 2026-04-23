#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above 12h Donchian upper band AND volume > 1.5x 20-period average AND 12h EMA50 > EMA100.
Short when price breaks below 12h Donchian lower band AND volume > 1.5x 20-period average AND 12h EMA50 < EMA100.
Exit when price retraces to 12h Donchian midpoint or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts, 
12h EMA trend filter to align with higher timeframe momentum, and ATR stops to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels: 20-period high/low
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_12h = (upper_12h + lower_12h) / 2.0
    
    # Align 12h Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    mid_aligned = align_htf_to_ltf(prices, df_12h, mid_12h)
    
    # 12h EMA trend filter: EMA50 and EMA100
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema100_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 100, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(ema100_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        mid_val = mid_aligned[i]
        ema50_val = ema50_aligned[i]
        ema100_val = ema100_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper band AND volume spike AND 12h EMA50 > EMA100 (uptrend)
            if (price > upper_val and volume[i] > 1.5 * vol_ma_val and ema50_val > ema100_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 12h Donchian lower band AND volume spike AND 12h EMA50 < EMA100 (downtrend)
            elif (price < lower_val and volume[i] > 1.5 * vol_ma_val and ema50_val < ema100_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 12h Donchian midpoint
            if position == 1 and price <= mid_val:
                exit_signal = True
            elif position == -1 and price >= mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12h_VolumeConfirmation_EMATrendFilter_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0