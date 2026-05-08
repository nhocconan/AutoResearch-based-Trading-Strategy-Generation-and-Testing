#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation (1.5x)
# In trending markets, price breaks Donchian channels with momentum; in ranging markets,
# the EMA filter prevents false breakouts. Volume confirms institutional participation.
# Target: 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.
# Works in bull (breakouts with trend) and bear (breakouts against trend filtered by EMA).

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = high_roll[i]
        lower = low_roll[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above upper band, above EMA50, volume confirmation
            if close[i] > upper and close[i] > ema50 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, below EMA50, volume confirmation
            elif close[i] < lower and close[i] < ema50 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band or below EMA50
            if close[i] < lower or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band or above EMA50
            if close[i] > upper or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals