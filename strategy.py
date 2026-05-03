#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R4/S4 breakout with 1w pivot filter and volume confirmation
# Long when price breaks above Camarilla R4 with volume > 1.8x 20-bar average AND weekly close > weekly pivot
# Short when price breaks below Camarilla S4 with volume > 1.8x 20-bar average AND weekly close < weekly pivot
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR, short exit when price > lowest_low_since_entry + 2.5 * ATR
# Using 1d timeframe as specified, targeting 30-100 trades over 4 years (7-25/year)
# Discrete sizing 0.25 to minimize fee drag, ATR stop with wider multiplier for 1d volatility

name = "1d_Camarilla_R4S4_1wPivot_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels (same timeframe, but needed for calculations)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (R4, S4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4 = pivot + range_1d * 1.1 / 2
    s4 = pivot - range_1d * 1.1 / 2
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 1w data ONCE before loop for weekly pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w pivot (weekly) and weekly close for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate ATR(20) for stoploss (wider for 1d volatility)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 20) + 1  # volume MA(20) + ATR(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(weekly_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R4 with volume spike AND weekly close > weekly pivot
            if (close[i] > r4_aligned[i] and 
                volume_spike[i] and weekly_close_aligned[i] > pivot_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Camarilla S4 with volume spike AND weekly close < weekly pivot
            elif (close[i] < s4_aligned[i] and 
                  volume_spike[i] and weekly_close_aligned[i] < pivot_1w_aligned[i]):
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