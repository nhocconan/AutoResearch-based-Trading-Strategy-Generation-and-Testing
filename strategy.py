#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d EMA Trend + Volume Spike + ATR Stoploss
Hypothesis: Breakouts of Donchian(20) channel in direction of 1d EMA(50) trend,
confirmed by volume spikes, work in both bull and bear markets by capturing
strong momentum moves. Volume ensures breakout validity, ATR stoploss manages risk.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Volume spike filter
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donch_upper[i]) or
            np.isnan(donch_lower[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # 1d trend: above EMA50 = uptrend, below = downtrend
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: close below Donchian lower OR stoploss
            if (close[i] <= donch_lower[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: close above Donchian upper OR stoploss
            if (close[i] >= donch_upper[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout in trend direction + volume spike
            long_breakout = (close[i] > donch_upper[i] and uptrend and vol_spike[i])
            short_breakout = (close[i] < donch_lower[i] and downtrend and vol_spike[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals