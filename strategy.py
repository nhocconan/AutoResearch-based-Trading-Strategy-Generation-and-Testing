#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Long when Alligator jaws < teeth < lips (bullish alignment) with volume > 2.0x 24-bar average and CHOP > 61.8 (range)
# Short when Alligator jaws > teeth > lips (bearish alignment) with volume > 2.0x 24-bar average and CHOP > 61.8 (range)
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR, short exit when price > lowest_low_since_entry + 2.5 * ATR
# Williams Alligator identifies trend alignment, volume confirms conviction, chop filter ensures ranging market for mean reversion.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee drag.

name = "12h_WilliamsAlligator_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR and chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on 1d for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Williams Alligator on 12h data
    # Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
    def smma(arr, period):
        # Smoothed Moving Average: first value is SMA, then recursive
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price := (high + low) / 2, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Calculate Choppiness Index on 1d
    def true_range(h, l, c):
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        return np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = pd.Series(true_range(high_1d, low_1d, close_1d)).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 24, 14) + 1  # SMMA periods + volume MA + ATR + shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment (jaws < teeth < lips) with volume spike and chop > 61.8 (range)
            if (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i] and 
                volume_spike[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: Alligator bearish alignment (jaws > teeth > lips) with volume spike and chop > 61.8 (range)
            elif (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i] and 
                  volume_spike[i] and chop_aligned[i] > 61.8):
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
            if close[i] < highest_since_entry - 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals