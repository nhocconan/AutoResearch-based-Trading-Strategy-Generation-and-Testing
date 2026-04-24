#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter + ATR-based stoploss.
- Primary timeframe: 4h for entries/exits, HTF: 12h for EMA trend.
- Donchian breakout: Long when price > highest high of last 20 bars, Short when price < lowest low of last 20 bars.
- Volume confirmation: current volume > 1.5 * 20-bar volume MA.
- Trend filter: only take longs when price > 12h EMA50, shorts when price < 12h EMA50.
- Stoploss: exit long when price drops below highest high since entry - 2.5 * ATR(20), exit short when price rises above lowest low since entry + 2.5 * ATR(20).
- Uses discrete signal size 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-bar volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0  # highest high since entry for longs
    entry_low = 0.0   # lowest low since entry for shorts
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_high = 0.0
                entry_low = 0.0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: price > highest high of last 20 bars with uptrend
                if close[i] > highest_high[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_high = high[i]  # initialize tracking high
                # Short breakdown: price < lowest low of last 20 bars with downtrend
                elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_low = low[i]  # initialize tracking low
        elif position == 1:
            # Update highest high since entry
            entry_high = max(entry_high, high[i])
            # ATR-based stoploss: exit if price drops below entry_high - 2.5 * ATR
            if close[i] < entry_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_high = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            entry_low = min(entry_low, low[i])
            # ATR-based stoploss: exit if price rises above entry_low + 2.5 * ATR
            if close[i] > entry_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_low = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0