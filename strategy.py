#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 12h EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND close < 12h EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses the Donchian middle band (20-period SMA of high/low).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
The 12h EMA50 provides a robust trend filter that works in both bull and bear markets by avoiding counter-trend entries.
Volume confirmation filter set at 1.5x to balance signal quality and trade frequency for 6h timeframe.
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
    
    # Load 6h data for price action - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels on 6h data
    # Upper band: 20-period high
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period SMA of (high+low)/2
    hl_avg = (high_6h + low_6h) / 2
    donchian_middle = pd.Series(hl_avg).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels to 6h timeframe (already on 6h, but align for safety)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 12h EMA50 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND close < 12h EMA50 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian middle band
                if price < donchian_middle_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian middle band
                if price > donchian_middle_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0