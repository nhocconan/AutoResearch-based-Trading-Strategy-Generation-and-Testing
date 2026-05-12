#!/usr/bin/env python3
"""
12h_Supertrend_WeeklyTrend_4hVolumeSpike
Hypothesis: Supertrend on 4h captures medium-term direction, weekly trend filter ensures alignment with major cycles, and 4h volume spikes confirm breakout strength. Works in bull by following uptrend, in bear by following downtrend. Weekly filter reduces whipsaw in ranging markets.
"""

name = "12h_Supertrend_WeeklyTrend_4hVolumeSpike"
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

    # Get 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Supertrend on 4h: ATR(10) * 3
    atr_period = 10
    multiplier = 3
    
    # Calculate ATR
    tr1 = pd.Series(df_4h['high']).rolling(window=2).max() - pd.Series(df_4h['low']).rolling(window=2).min()
    tr2 = abs(pd.Series(df_4h['high']) - pd.Series(df_4h['close']).shift(1))
    tr3 = abs(pd.Series(df_4h['low']) - pd.Series(df_4h['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    
    # Basic upper and lower bands
    basic_ub = (df_4h['high'] + df_4h['low']) / 2 + multiplier * atr
    basic_lb = (df_4h['high'] + df_4h['low']) / 2 - multiplier * atr
    
    # Final upper and lower bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    
    for i in range(1, len(basic_ub)):
        if basic_ub[i] < final_ub[i-1] or df_4h['close'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or df_4h['close'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend direction
    supertrend = np.zeros(len(df_4h))
    for i in range(len(df_4h)):
        if i == 0:
            supertrend[i] = 1  # Start with uptrend
        else:
            if close_4h := df_4h['close'].iloc[i] > final_ub[i-1]:
                supertrend[i] = 1
            elif close_4h < final_lb[i-1]:
                supertrend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                if supertrend[i] == 1 and final_lb[i] < final_lb[i-1]:
                    final_lb[i] = final_lb[i-1]
                if supertrend[i] == -1 and final_ub[i] > final_ub[i-1]:
                    final_ub[i] = final_ub[i-1]
    
    # Align Supertrend to 12h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)

    # Weekly trend: EMA50 slope
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_slope = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    weekly_uptrend = ema_50_1w_slope > 0
    weekly_downtrend = ema_50_1w_slope < 0
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))

    # Volume spike on 4h: >2.0x 20-period average
    vol_ma_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = df_4h['volume'].values > (2.0 * vol_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(supertrend_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(volume_spike_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Supertrend uptrend + weekly uptrend + volume spike
            if (supertrend_aligned[i] == 1 and 
                weekly_uptrend_aligned[i] > 0.5 and 
                volume_spike_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend + weekly downtrend + volume spike
            elif (supertrend_aligned[i] == -1 and 
                  weekly_downtrend_aligned[i] > 0.5 and 
                  volume_spike_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend turns down
            if supertrend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend turns up
            if supertrend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals