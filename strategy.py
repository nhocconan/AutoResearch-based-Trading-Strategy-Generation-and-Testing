#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chandelier Exit with volume confirmation and 1d trend filter
# Uses Chandelier Exit (ATR-based trailing stop) to capture trends in both bull and bear markets
# Only trades when volume confirms breakout and higher timeframe trend aligns
# Targets 20-50 trades per year to minimize fee drag
name = "4h_ChandelierExit_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h ATR for Chandelier Exit
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    
    # Chandelier Exit: 22-period ATR multiplier of 3.0
    # Long exit: highest high - 3*ATR
    # Short exit: lowest low + 3*ATR
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low).rolling(window=22, min_periods=22).min().values
    chandelier_long_exit = highest_high - 3.0 * atr_4h
    chandelier_short_exit = lowest_low + 3.0 * atr_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(atr_4h[i]) or np.isnan(chandelier_long_exit[i]) or np.isnan(chandelier_short_exit[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: price above Chandelier long exit + volume + 1d uptrend
            if price > chandelier_long_exit[i] and volume_filter and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Chandelier short exit + volume + 1d downtrend
            elif price < chandelier_short_exit[i] and volume_filter and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Chandelier long exit
            if price < chandelier_long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Chandelier short exit
            if price > chandelier_short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals