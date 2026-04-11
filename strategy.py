#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakout with daily volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels (derived from prior day's range) act as strong support/resistance.
# Breakouts above H4 or below L4 with volume confirmation and daily EMA50 trend filter.
# Works in bull (breakouts up) and bear (breakouts down) markets. Low frequency to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # These are the key breakout levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i]
        breakout_down = close[i] < camarilla_l4_aligned[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if breakout_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: breakout failure or trend change
        elif position == 1 and (not breakout_up or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not breakout_down or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals