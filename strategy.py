#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator for trend direction and 4h price action for entry timing.
# Long when: 4h close > 1d Alligator Jaw (uptrend) AND 4h close > 4h Donchian(20) upper band with volume > 1.5x 20-bar average
# Short when: 4h close < 1d Alligator Jaw (downtrend) AND 4h close < 4h Donchian(20) lower band with volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1d Williams Alligator for robust trend filtering (avoids whipsaw), 4h Donchian breakouts for entry precision, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Alligator1d_Donchian20_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator (Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) function
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines
    jaw_1d = smma(close_1d, 13)  # Jaw (Blue) - 13-period SMMA
    teeth_1d = smma(close_1d, 8)  # Teeth (Red) - 8-period SMMA
    lips_1d = smma(close_1d, 5)   # Lips (Green) - 5-period SMMA
    
    # Align 1d Alligator Jaw to 4h timeframe (completed 1d bar only)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    
    # 4h Donchian(20) for entry timing
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Donchian and ATR calculations)
    start_idx = max(donchian_window, 30) + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: 1d Alligator Jaw uptrend (price > jaw) AND price breaks above 4h Donchian upper with volume spike
            if close[i] > jaw_1d_aligned[i] and close[i] > upper_channel[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: 1d Alligator Jaw downtrend (price < jaw) AND price breaks below 4h Donchian lower with volume spike
            elif close[i] < jaw_1d_aligned[i] and close[i] < lower_channel[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals