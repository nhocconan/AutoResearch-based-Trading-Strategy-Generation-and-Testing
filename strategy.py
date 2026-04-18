#!/usr/bin/env python3
"""
1h 4h/1d Trend + Volume Spike with Session Filter
Hypothesis: Trend following on 1h using 4h EMA50 direction and 1d EMA200 filter,
with volume confirmation (1.5x 20-bar average) and session filter (08-20 UTC).
This captures momentum with institutional participation while avoiding low-volume noise.
Designed for low trade frequency: ~20-40 trades/year.
"""

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
    
    # Get 4h data for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for long-term filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for long-term trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        trend_4h = ema50_4h_aligned[i]
        trend_1d = ema200_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price above both trends + volume spike
            if close[i] > trend_4h and close[i] > trend_1d and vol_ok:
                signals[i] = 0.20
                position = 1
            # Enter short: price below both trends + volume spike
            elif close[i] < trend_4h and close[i] < trend_1d and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below 4h trend or volume spike in opposite direction
            if close[i] < trend_4h or (volume[i] > vol_ma[i] * 2.0 and close[i] < close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 4h trend or volume spike in opposite direction
            if close[i] > trend_4h or (volume[i] > vol_ma[i] * 2.0 and close[i] > close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Volume_Spike_Session"
timeframe = "1h"
leverage = 1.0