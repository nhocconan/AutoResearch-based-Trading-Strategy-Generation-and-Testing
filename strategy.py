#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 levels from daily timeframe act as support/resistance in ranging markets.
Breakout above R1 with daily uptrend and volume confirmation signals long.
Breakdown below S1 with daily downtrend and volume confirmation signals short.
Uses 1h volume spike to avoid false breakouts. Works in both bull/bear regimes by following daily trend.
Target: 20-40 trades/year per symbol.
"""

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily high/low/close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 4h timeframe (wait for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = df_1d['close'].values > ema_34_1d
    downtrend_1d = df_1d['close'].values < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume spike: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: break above R1, daily uptrend, volume spike
            if close[i] > R1_aligned[i] and uptrend_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, daily downtrend, volume spike
            elif close[i] < S1_aligned[i] and downtrend_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below S1 or loss of daily uptrend
            if close[i] < S1_aligned[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above R1 or loss of daily downtrend
            if close[i] > R1_aligned[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals