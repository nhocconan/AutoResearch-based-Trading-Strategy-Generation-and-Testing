#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Only long when price breaks above R1 and close > 1d EMA34, short when breaks below S1 and close < 1d EMA34.
Volume spike (>1.5x 20-period EMA volume) confirms institutional interest.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
Camarilla pivot levels provide intraday support/resistance that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume (12h timeframe)
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup for volume EMA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # Camarilla uses previous day's high, low, close
        if i >= 2:  # Need at least 2 bars for previous day (12h timeframe = 2 bars per day)
            prev_high = high[i-2]  # Previous day high
            prev_low = low[i-2]    # Previous day low
            prev_close = close[i-2] # Previous day close
            
            # Camarilla R1, S1 levels
            rang = prev_high - prev_low
            r1 = prev_close + rang * 1.1 / 12
            s1 = prev_close - rang * 1.1 / 12
            
            # Long logic: break above R1 + close > 1d EMA34 + volume spike
            if close[i] > r1 and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short logic: break below S1 + close < 1d EMA34 + volume spike
            elif close[i] < s1 and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: price returns to previous day's close (mean reversion to midpoint)
            elif position == 1 and close[i] < prev_close:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > prev_close:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        else:
            # Not enough data for Camarilla calculation, hold flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0