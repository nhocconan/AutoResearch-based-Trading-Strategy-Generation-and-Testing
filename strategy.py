#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout_TrendVol_v2
Hypothesis: 12-hour breakouts from daily Camarilla R1/S1 levels with daily EMA50 trend filter and volume spike confirmation.
Only takes long when price breaks above R1 with volume spike and daily uptrend, short when breaks below S1 with volume spike and daily downtrend.
Uses tighter entry conditions (requires volume spike >2.5x 30-period average and momentum confirmation via 3-period ROC > 0) to reduce trade frequency.
Targets 20-40 trades per year on 12h timeframe to avoid overtrading and fee drag.
Works in bull markets via trend-following breaks and in bear markets via counter-trend reversals at extreme daily levels.
"""

name = "12h_1D_Camarilla_R1_S1_Breakout_TrendVol_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.5x 30-period average (more selective than original 2.0x 20-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Momentum confirmation: 3-period ROC > 0 (avoids chop)
    roc = np.zeros_like(close)
    roc[3:] = (close[3:] - close[:-3]) / close[:-3]
    momentum_up = roc > 0
    momentum_down = roc < 0
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.0 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.0 / 4
    
    # Align daily levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + daily uptrend + upward momentum
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i] and
                momentum_up[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + daily downtrend + downward momentum
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  momentum_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below daily EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above daily EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals