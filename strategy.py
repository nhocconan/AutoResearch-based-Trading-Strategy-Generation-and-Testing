#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
- Primary timeframe: 12h (target 12-37 trades/year, 50-150 over 4 years)
- Entry: Price breaks Donchian(20) high/low with volume > 1.5x 20-period average
- Trend filter: Only trade breakouts in direction of 1d ATR trend (rising ATR = volatile/trending market)
- Exit: Opposite Donchian(10) breakout or ATR contraction (< 0.8x 20-period ATR average)
- Works in bull/bear markets by capturing volatility expansion breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian channels on 12h data
    lookback = 20
    exit_lookback = 10
    
    # Upper/lower Donchian(20) for entry
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Upper/lower Donchian(10) for exit
    donch_high_exit = pd.Series(high).rolling(window=exit_lookback, min_periods=exit_lookback).max().values
    donch_low_exit = pd.Series(low).rolling(window=exit_lookback, min_periods=exit_lookback).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR trend filter: rising ATR (> 0.5x 20-period ATR MA) indicates trending/volatile market
    atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 14)  # for Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(donch_high_exit[i]) or np.isnan(donch_low_exit[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian(20) high + volume + rising ATR
            long_breakout = (close[i] > donch_high[i-1] and 
                           volume[i] > 1.5 * vol_ma[i] and
                           atr_14_1d_aligned[i] > 0.5 * atr_ma[i])
            
            # Short breakout: price < Donchian(20) low + volume + rising ATR
            short_breakout = (close[i] < donch_low[i-1] and 
                            volume[i] > 1.5 * vol_ma[i] and
                            atr_14_1d_aligned[i] > 0.5 * atr_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price < Donchian(10) low OR ATR contraction (< 0.8x ATR MA)
                if (close[i] < donch_low_exit[i-1]) or \
                   (atr_14_1d_aligned[i] < 0.8 * atr_ma[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price > Donchian(10) high OR ATR contraction
                if (close[i] > donch_high_exit[i-1]) or \
                   (atr_14_1d_aligned[i] < 0.8 * atr_ma[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0