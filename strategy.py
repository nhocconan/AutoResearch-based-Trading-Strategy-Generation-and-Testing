#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# 1d EMA50 ensures trades align with higher timeframe direction (avoid counter-trend)
# Volume spike (>1.8x 30-period EMA volume) confirms institutional participation
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: 30-period EMA of volume
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ema_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND 1d EMA50 uptrend AND volume spike
            if close[i] > upper[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_30[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND 1d EMA50 downtrend AND volume spike
            elif close[i] < lower[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_30[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below upper Donchian OR 1d EMA50 turns down
            if close[i] < upper[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above lower Donchian OR 1d EMA50 turns up
            if close[i] > lower[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals