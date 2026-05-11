#!/usr/bin/env python3
"""
4h_DailyPivot_Breakout_Trend_Volume
Hypothesis: Price breaking above/below daily pivot-derived support/resistance (R1/S1) with
10-period momentum confirmation, volume spike, and weekly trend filter (price > weekly EMA50).
Daily pivots capture short-term institutional levels; momentum confirms breakout strength;
volume validates participation; weekly trend avoids counter-trend whipsaws. Designed for
low frequency (~25-40 trades/year) to work in both bull (breakouts) and bear (mean reversion
at extremes) markets.
"""

name = "4h_DailyPivot_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for pivot calculation (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Daily Pivot Points (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point
    pp_d = (high_1d + low_1d + close_1d) / 3.0
    # Daily R1 and S1
    r1_d = 2 * pp_d - low_1d
    s1_d = 2 * pp_d - high_1d
    
    # Align daily levels to 4h timeframe (using previous day's levels)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- 10-period Momentum (rate of change) ---
    mom = np.zeros_like(close)
    mom[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(mom[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: Price > daily R1 AND positive momentum AND volume spike AND above weekly EMA50
        # Short: Price < daily S1 AND negative momentum AND volume spike AND below weekly EMA50
        long_entry = (close[i] > r1_d_aligned[i]) and \
                     (mom[i] > 0) and \
                     vol_spike[i] and \
                     (close[i] > ema_50_1w_aligned[i])
        
        short_entry = (close[i] < s1_d_aligned[i]) and \
                      (mom[i] < 0) and \
                      vol_spike[i] and \
                      (close[i] < ema_50_1w_aligned[i])
        
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
            # Long: Price crosses below daily pivot OR momentum turns negative OR below weekly EMA50
            # Short: Price crosses above daily pivot OR momentum turns positive OR above weekly EMA50
            if position == 1:
                pp_d_aligned = align_htf_to_ltf(prices, df_1d, pp_d)
                if (close[i] < pp_d_aligned[i]) or \
                   (mom[i] < 0) or \
                   (close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                pp_d_aligned = align_htf_to_ltf(prices, df_1d, pp_d)
                if (close[i] > pp_d_aligned[i]) or \
                   (mom[i] > 0) or \
                   (close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals