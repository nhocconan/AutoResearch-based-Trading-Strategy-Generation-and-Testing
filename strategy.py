#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 1d EMA Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves, volume confirms institutional participation,
and 1d EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
Designed for low trade frequency (20-40/year) to minimize fee drag while capturing sustained moves.
Works in both bull (long breakouts) and bear (short breakdowns) markets.
"""
name = "4h_Donchian_Breakout_Volume_1dEMA"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # === 1d EMA (34) for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + volume spike + above 1d EMA (uptrend)
            if (close[i] > highest_high[i] and 
                vol_spike[i] and
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower + volume spike + below 1d EMA (downtrend)
            elif (close[i] < lowest_low[i] and 
                  vol_spike[i] and
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower (reversal signal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper (reversal signal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals