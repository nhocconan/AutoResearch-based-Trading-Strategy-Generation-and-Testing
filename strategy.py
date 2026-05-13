#!/usr/bin/env python3
"""
4h_Pivot_Reversal_Strategy
Hypothesis: Daily pivot point reversals with volume confirmation and ADX trend filter capture mean reversion in both bull and bear markets.
Pivots act as institutional support/resistance. Works in ranging markets (ADX < 25) and captures reversals in trends (ADX > 25).
Designed for low trade frequency (20-40/year) with clear entry/exit rules.
"""

name = "4h_Pivot_Reversal_Strategy"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ADX(14) for trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get daily pivot points (standard formula)
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day's OHLC
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    r2 = pp + (df_1d['high'] - df_1d['low'])
    s2 = pp - (df_1d['high'] - df_1d['low'])
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price touches or crosses below S1 with volume confirmation
            if low[i] <= s1_aligned[i] and volume_confirm[i]:
                # Additional filters: ADX < 30 (not strong trend) OR price above PP (mean reversion bias)
                if adx[i] < 30 or close[i] > pp_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price touches or crosses above R1 with volume confirmation
            elif high[i] >= r1_aligned[i] and volume_confirm[i]:
                # Additional filters: ADX < 30 (not strong trend) OR price below PP (mean reversion bias)
                if adx[i] < 30 or close[i] < pp_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches PP or R1, or ADX > 35 (strong trend emerging)
            if high[i] >= pp_aligned[i] or high[i] >= r1_aligned[i] or adx[i] > 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches PP or S1, or ADX > 35 (strong trend emerging)
            if low[i] <= pp_aligned[i] or low[i] <= s1_aligned[i] or adx[i] > 35:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals