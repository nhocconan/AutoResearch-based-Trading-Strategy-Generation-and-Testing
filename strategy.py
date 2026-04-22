# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Hypothesis: 4-hour ATR breakout with daily volatility filter and volume confirmation.
This strategy captures volatility expansion after periods of low volatility, using
daily ATR ratio to filter for high-probability breakout environments. Volume spikes
confirm institutional participation. Works in both bull and bear markets by
trading breakouts in the direction of the breakout, with volatility-based stops.
Target: 25-40 trades/year per symbol.
"""

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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(10) and ATR(30) for volatility ratio
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr10_1d = tr_1d.rolling(window=10, min_periods=10).mean().values
    atr30_1d = tr_1d.rolling(window=30, min_periods=30).mean().values
    
    # ATR ratio: current volatility vs longer-term average
    atr_ratio = atr10_1d / atr30_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4-day high/low for breakout levels
    high_4d = high_1d.rolling(window=4, min_periods=4).max().values
    low_4d = low_1d.rolling(window=4, min_periods=4).min().values
    high_4d_aligned = align_htf_to_ltf(prices, df_1d, high_4d)
    low_4d_aligned = align_htf_to_ltf(prices, df_1d, low_4d)
    
    # Calculate 4h ATR for stop loss
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = high[0] - low[0]
    tr_4h2[0] = np.abs(high[0] - close[0])
    tr_4h3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(high_4d_aligned[i]) or 
            np.isnan(low_4d_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 4-day high with volatility expansion and volume spike
            if (close[i] > high_4d_aligned[i] and 
                atr_ratio_aligned[i] > 1.2 and  # Volatility expanding
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Break below 4-day low with volatility expansion and volume spike
            elif (close[i] < low_4d_aligned[i] and 
                  atr_ratio_aligned[i] > 1.2 and  # Volatility expanding
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # ATR-based trailing stop
            stop_signal = False
            
            if position == 1:
                # Trail stop: exit if price drops 2*ATR from highest high since entry
                # Simplified: exit if close < (entry high - 2*ATR)
                # We'll use a trailing high based on recent highs
                recent_high = np.maximum.accumulate(high[max(0, i-20):i+1])[-1]
                if close[i] < recent_high - 2.0 * atr_4h[i]:
                    stop_signal = True
            else:  # position == -1
                # Trail stop: exit if price rises 2*ATR from lowest low since entry
                recent_low = np.minimum.accumulate(low[max(0, i-20):i+1])[-1]
                if close[i] > recent_low + 2.0 * atr_4h[i]:
                    stop_signal = True
            
            if stop_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ATR_Breakout_Volatility_Volume"
timeframe = "4h"
leverage = 1.0