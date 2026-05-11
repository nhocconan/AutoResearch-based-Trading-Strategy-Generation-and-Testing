#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_Breakout_1D_EMA34_Trend
Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with volume confirmation, 
filtered by daily trend (price > EMA34). Daily pivots capture institutional levels; 
breakouts signal momentum; volume confirms participation. Daily trend filter avoids 
counter-trend whipsaws. Designed for low frequency (12-37 trades/year) to work in 
both bull (breakouts) and bear (mean reversion at extremes) markets.
"""

name = "12h_1D_Camarilla_R1S1_Breakout_1D_EMA34_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
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
    # Daily R1 and S1 (Camarilla)
    r1_d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    s1_d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align daily levels to 12h timeframe (using previous day's levels)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # --- Daily EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma.values[i] if i < len(vol_ma.values) else np.nan)):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: Price > daily R1 AND volume spike AND above daily EMA34
        # Short: Price < daily S1 AND volume spike AND below daily EMA34
        long_entry = (close[i] > r1_d_aligned[i]) and \
                     vol_spike[i] and \
                     (close[i] > ema_34_1d_aligned[i])
        
        short_entry = (close[i] < s1_d_aligned[i]) and \
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
            # Long: Price crosses below daily pivot OR below daily EMA34
            # Short: Price crosses above daily pivot OR above daily EMA34
            if position == 1:
                pp_d_aligned = align_htf_to_ltf(prices, df_1d, pp_d)
                if (close[i] < pp_d_aligned[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                pp_d_aligned = align_htf_to_ltf(prices, df_1d, pp_d)
                if (close[i] > pp_d_aligned[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals