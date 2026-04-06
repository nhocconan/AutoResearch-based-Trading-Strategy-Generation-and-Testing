#!/usr/bin/env python3
"""
1d Donchian(20) breakout + EMA50 trend filter + Volume confirmation.
Hypothesis: Price breaks above/below 20-day Donchian channel with EMA50 trend and volume confirmation captures strong moves in both bull and bear markets. Low trade frequency reduces fee drag.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14318_1d_donchian20_ema50_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for regime filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    donch_period = 20
    upper_dc = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_dc = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donch_period, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to lower Donchian OR stoploss
            if close[i] <= lower_dc[i] or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to upper Donchian OR stoploss
            if close[i] >= upper_dc[i] or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA50 trend + volume
            long_breakout = close[i] > upper_dc[i]
            short_breakout = close[i] < lower_dc[i]
            
            # EMA50 trend filter: price above EMA50 for long, below for short
            ema_filter_long = close[i] > ema_50[i]
            ema_filter_short = close[i] < ema_50[i]
            
            # Weekly trend filter: only trade in direction of weekly trend
            weekly_filter_long = close[i] > ema_1w_aligned[i]
            weekly_filter_short = close[i] < ema_1w_aligned[i]
            
            if long_breakout and ema_filter_long and weekly_filter_long and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and ema_filter_short and weekly_filter_short and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals