#!/usr/bin/env python3
# 12h_1W_1D_Camarilla_R1_S1_Breakout
# Hypothesis: Breakouts from weekly and daily Camarilla R1/S1 levels on 12h timeframe, filtered by weekly trend (price > weekly EMA50) and confirmed by volume spikes.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion from extremes (short at S1, long at R1).
# Uses weekly and daily timeframe for structure and trend, 12h for execution to limit trade frequency.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_1W_1D_Camarilla_R1_S1_Breakout"
timeframe = "12h"
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
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for trend filter and weekly Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly data for Camarilla R1/S1 levels
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    rang_1w = prev_high_1w - prev_low_1w
    R1_1w = prev_close_1w + 1.1 * rang_1w * 1.1 / 4
    S1_1w = prev_close_1w - 1.1 * rang_1w * 1.1 / 4
    
    # Align weekly Camarilla levels to 12h timeframe
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # Daily data for daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily data for Camarilla R1/S1 levels
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.1 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        if (np.isnan(R1_1w_aligned[i]) or np.isnan(S1_1w_aligned[i]) or
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above BOTH weekly and daily R1 + volume spike + price above weekly EMA50 (uptrend)
            if (close[i] > R1_1w_aligned[i] and close[i] > R1_1d_aligned[i] and
                volume_spike[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below BOTH weekly and daily S1 + volume spike + price below weekly EMA50 (downtrend)
            elif (close[i] < S1_1w_aligned[i] and close[i] < S1_1d_aligned[i] and
                  volume_spike[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters either weekly or daily H-L range OR closes below weekly EMA50
            if ((close[i] < R1_1w_aligned[i] and close[i] > S1_1w_aligned[i]) or
                (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters either weekly or daily H-L range OR closes above weekly EMA50
            if ((close[i] < R1_1w_aligned[i] and close[i] > S1_1w_aligned[i]) or
                (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals