#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1DTrend_Volume
Hypothesis: Uses daily trend filter (EMA34) to determine direction, then takes Donchian(20) breakouts on 12h with volume confirmation.
Works in bull markets (breakouts with trend) and bear markets (mean reversion during trend reversals) by filtering counter-trend trades.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "12h_Donchian_Breakout_1DTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Donchian(20) channels ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume confirmation (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_long = (high[i] > highest_high[i-1]) and vol_spike[i]
        breakout_short = (low[i] < lowest_low[i-1]) and vol_spike[i]
        
        if position == 0:
            # Only take trades in direction of trend
            if breakout_long and uptrend:
                signals[i] = 0.25
                position = 1
            elif breakout_short and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit on opposite breakout or trend reversal
            if position == 1:
                exit_signal = breakout_short or (not uptrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                exit_signal = breakout_long or (not downtrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals