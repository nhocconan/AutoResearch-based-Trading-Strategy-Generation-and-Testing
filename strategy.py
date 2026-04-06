#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Price breaks above/below Donchian(20) channel with alignment to weekly pivot direction and volume surge capture institutional moves in both bull and bear markets. Weekly pivot provides structural bias, Donchian captures breakouts, volume confirms conviction.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h and 1d data for weekly pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly pivot points from prior week (using daily OHLC)
    # Calculate weekly high/low/close from daily data (simplified: use 5-day aggregation)
    # We'll approximate weekly pivot using daily high/low/close of the week
    # For simplicity, use prior day's OHLC for pivot (common intraday technique)
    # In practice, would use weekly OHLC, but we approximate with daily for available data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Classic pivot point: (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance levels: R1 = 2*PP - L, R2 = PP + (H - L), R3 = H + 2*(PP - L)
    r1 = 2 * pp - low_1d
    r2 = pp + (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    # Support levels: S1 = 2*PP - H, S2 = PP - (H - L), S3 = L - 2*(H - PP)
    s1 = 2 * pp - high_1d
    s2 = pp - (high_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')  # placeholder, will use pandas
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR(14) for stoploss
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
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below S3 OR stoploss
            if (close[i] <= s3_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above R3 OR stoploss
            if (close[i] >= r3_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with pivot direction and volume
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # Weekly pivot bias: price above PP = bullish bias, below PP = bearish bias
            bullish_bias = close[i] > pp_aligned[i]
            bearish_bias = close[i] < pp_aligned[i]
            
            if long_breakout and bullish_bias and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and bearish_bias and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals