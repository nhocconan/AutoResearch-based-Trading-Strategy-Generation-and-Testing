#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when price crosses the Donchian middle band (20-period SMA of high/low).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
The daily EMA34 provides a robust trend filter that works in both bull and bear markets by avoiding counter-trend entries.
Volume confirmation filter set high (1.8x) to reduce false breakouts and overtrading.
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
    
    # Load 4h data for price action - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels on 4h data
    # Upper band: 20-period high
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period SMA of (high+low)/2
    hl_avg = (high_4h + low_4h) / 2
    donchian_middle = pd.Series(hl_avg).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels to 4h timeframe (already on 4h, but align for safety)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 1d EMA34 AND volume spike
            if (price > donchian_upper_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND close < 1d EMA34 AND volume spike
            elif (price < donchian_lower_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
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

name = "4H_Donchian20_1dEMA34_VolumeConfirm_Strict"
timeframe = "4h"
leverage = 1.0