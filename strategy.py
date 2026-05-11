#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Price breaking above/below daily Camarilla pivot levels (R1/S1) with 
4h Donchian(20) breakout in same direction, volume spike confirmation, and daily EMA34 trend filter.
Camarilla pivots from daily timeframe provide institutional short-term support/resistance.
Donchian breakout captures momentum; volume spike confirms institutional participation.
Daily EMA34 filter ensures trading in direction of intermediate trend to avoid counter-trend whipsaws.
Designed for low frequency (20-40 trades/year) to work in both bull (breakouts with trend) and bear 
(mean reversion at extreme pivots with trend filter preventing false signals) markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivot Levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point
    pp_d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla R1 and S1
    r1_d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily levels to 4h timeframe (using previous day's levels)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # --- Daily EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Donchian Channel (20) on 4h ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: Price > daily R1 AND breaks Donchian high AND volume spike AND above daily EMA34
        # Short: Price < daily S1 AND breaks Donchian low AND volume spike AND below daily EMA34
        long_entry = (close[i] > r1_d_aligned[i]) and \
                     (high[i] > highest_high[i-1]) and \
                     vol_spike[i] and \
                     (close[i] > ema_34_1d_aligned[i])
        
        short_entry = (close[i] < s1_d_aligned[i]) and \
                      (low[i] < lowest_low[i-1]) and \
                      vol_spike[i] and \
                      (close[i] < ema_34_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: 
            # Long: Price crosses below daily pivot point OR Donchian low OR below daily EMA34
            # Short: Price crosses above daily pivot point OR Donchian high OR above daily EMA34
            pp_d_aligned = align_htf_to_ltf(prices, df_1d, pp_d)
            if position == 1:
                if (close[i] < pp_d_aligned[i]) or \
                   (low[i] < lowest_low[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > pp_d_aligned[i]) or \
                   (high[i] > highest_high[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals