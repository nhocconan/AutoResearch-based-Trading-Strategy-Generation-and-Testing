#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: Trade breakouts of daily Donchian channels in direction of weekly trend.
Uses weekly EMA for trend direction, daily Donchian breakout for entry, volume spike for confirmation.
Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(20) for trend direction
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (volume > 1.5x 20-day EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema)
    
    # ATR(14) for stoploss
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
    start = 100
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend: above EMA20 = uptrend, below = downtrend
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donch_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donch_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout in trend direction + volume spike
            long_setup = (close[i] > donch_high[i] and uptrend and vol_spike[i])
            short_setup = (close[i] < donch_low[i] and downtrend and vol_spike[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals