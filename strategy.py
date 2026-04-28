#!/usr/bin/env python3
# Hypothesis: 4h Camarilla Pivot R4/S4 breakout with 1d volume spike and 1d EMA(34) trend filter.
# Uses daily Camarilla levels for major support/resistance, requiring price to break through
# R4 (strong resistance) or S4 (strong support) with volume confirmation (2x 20-day average).
# Trend filter uses 1d EMA(34) to ensure alignment with daily trend.
# Designed for low-frequency, high-conviction trades targeting 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R4 = C + (H-L) * 1.1/2, S4 = C - (H-L) * 1.1/2
    r4 = df_1d['close'] + (range_hl * 1.1 / 2)
    s4 = df_1d['close'] - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Volume filter: 1d volume > 2x 20-day average
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_20 * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: price breaks R4/S4 with volume spike
        long_breakout = (close[i] > r4_aligned[i]) and volume_spike_aligned[i]
        short_breakout = (close[i] < s4_aligned[i]) and volume_spike_aligned[i]
        
        # Exit conditions: price returns inside Camarilla H-L range
        typical_today = (high[i] + low[i] + close[i]) / 3
        range_today = high[i] - low[i]
        r4_today = close[i] + (range_today * 1.1 / 2)  # Approximate for exit
        s4_today = close[i] - (range_today * 1.1 / 2)
        
        long_exit = close[i] < r4_today
        short_exit = close[i] > s4_today
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0