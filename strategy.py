#!/usr/bin/env python3
# 6h_Aroon_BullBear_Trend_With_Volume
# Hypothesis: Uses Aroon Oscillator on 6h to detect early trend changes (strength of uptrend/downtrend), confirmed by 1d EMA50 trend and volume spike. Designed to capture trending moves in both bull and bear markets with low trade frequency to minimize fee drag. Aroon helps identify when a new trend is starting, reducing whipsaw in choppy markets.

timeframe = "6h"
name = "6h_Aroon_BullBear_Trend_With_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on daily closes
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Aroon Oscillator (25-period) on 6h data
    # Aroon Up = ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down = ((25 - periods since 25-period low) / 25) * 100
    # Aroon Oscillator = Aroon Up - Aroon Down
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low in the last 'period' bars
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        # Find index of max high and min low (most recent if tie)
        high_idx = i - np.argmax(window_high[::-1])  # argmax from reversed gives most recent
        low_idx = i - np.argmin(window_low[::-1])
        periods_since_high = i - high_idx
        periods_since_low = i - low_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100
    
    # Volume spike detection: 1.5x average volume (72-period = 3 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=72, min_periods=72).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(72, 50, period - 1)  # Ensure we have all data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(aroon_osc[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aroon > 50 (strong uptrend) + volume spike + 1d uptrend
            if aroon_osc[i] > 50 and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Aroon < -50 (strong downtrend) + volume spike + 1d downtrend
            elif aroon_osc[i] < -50 and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Aroon turns negative (trend weakness) or 1d trend fails
            if aroon_osc[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Aroon turns positive (trend weakness) or 1d trend fails
            if aroon_osc[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals