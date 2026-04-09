#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and ATR filter
# In trending regimes: breakout above/below Donchian(20) channels with volume > 1.5x average
# Uses ATR(14) for volatility normalization and stoploss
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: breakout catches momentum, volume filter avoids false signals

name = "12h_1d_donchian_breakout_volume_atr_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period) based on prior day to avoid look-ahead
    # Shift by 1 to use only completed daily bars
    high_shift = np.concatenate([[np.nan], high_1d[:-1]])
    low_shift = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Donchian upper: highest high over past 20 completed days
    donch_high_1d = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over past 20 completed days
    donch_low_1d = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume confirmation: current 12h volume > 1.5x average 1d volume
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(donch_high_1d_aligned[i]) or 
            np.isnan(donch_low_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or ATR-based stoploss
            if close[i] < donch_low_1d_aligned[i] or close[i] < donch_high_1d_aligned[i] - 2.0 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or ATR-based stoploss
            if close[i] > donch_high_1d_aligned[i] or close[i] > donch_low_1d_aligned[i] + 2.0 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation
            if close[i] > donch_high_1d_aligned[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume confirmation
            elif close[i] < donch_low_1d_aligned[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals