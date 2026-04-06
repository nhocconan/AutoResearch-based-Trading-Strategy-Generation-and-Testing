#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation.
# Enter long when price breaks above 4h Donchian upper with 1d EMA(50) rising and volume > 1.5x avg.
# Enter short when price breaks below 4h Donchian lower with 1d EMA(50) falling and volume > 1.5x avg.
# Exit on opposite Donchian breakout or when price crosses 1d EMA(50).
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_donchian20_1dema50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_asc = pd.Series(ema_50).diff() > 0  # Rising EMA
    ema_50_desc = pd.Series(ema_50).diff() < 0  # Falling EMA
    ema_50_asc_aligned = align_htf_to_ltf(prices, df_1d, ema_50_asc.values)
    ema_50_desc_aligned = align_htf_to_ltf(prices, df_1d, ema_50_desc.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR crosses below EMA50
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR crosses above EMA50
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA50 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donch_high[i] and ema_50_asc_aligned[i]:
                    # Breakout above Donchian in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low[i] and ema_50_desc_aligned[i]:
                    # Breakdown below Donchian in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals