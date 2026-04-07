#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based breakout with 1d trend filter and volume confirmation
# Uses ATR breakouts to capture momentum in trending markets while avoiding whipsaws
# 1d EMA100 filter ensures alignment with higher timeframe trend
# Volume confirmation filters out low-conviction breakouts
# Designed for low frequency (target: 20-40 trades/year) to minimize fee impact
# Works in both bull/bear via trend-following logic: only trade in direction of higher timeframe trend

name = "6h_atr_breakout_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # ATR calculation (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period breakout)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_100_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA100
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakdown_down = close[i] < lowest_low[i]
        
        # Exit conditions: close back inside the channel
        exit_long = close[i] < lowest_low[i]
        exit_short = close[i] > highest_high[i]
        
        if position == 1:  # Long position
            # Exit when price breaks below the lower channel
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when price breaks above the upper channel
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in direction of higher timeframe trend with volume confirmation
            if uptrend and vol_confirm and breakout_up:
                position = 1
                signals[i] = 0.25
            elif downtrend and vol_confirm and breakdown_down:
                position = -1
                signals[i] = -0.25
    
    return signals