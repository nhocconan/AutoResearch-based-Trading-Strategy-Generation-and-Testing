#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
Only trade breakouts in the direction of the 1-day EMA34 trend to avoid counter-trend trades.
Uses volume spike (2x 20-period average) to confirm breakout strength.
Designed for low trade frequency (20-50 trades/year) by requiring trend alignment, breakout, and volume confirmation.
Works in bull markets (follow uptrend) and bear markets (follow downtrend) via adaptive EMA34 trend filter.
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
    
    # Load 1d data for EMA34 trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = highest high of last 20 periods
    # Donchian lower = lowest low of last 20 periods
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Uptrend (EMA34 rising) + price breaks above Donchian upper + volume spike
            if ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and close[i] > upper_20_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (EMA34 falling) + price breaks below Donchian lower + volume spike
            elif ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and close[i] < lower_20_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend reversal or price returns to opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Exit long: Downtrend or price breaks below Donchian lower
                if ema34_1d_aligned[i] < ema34_1d_aligned[i-1] or close[i] < lower_20_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Uptrend or price breaks above Donchian upper
                if ema34_1d_aligned[i] > ema34_1d_aligned[i-1] or close[i] > upper_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0