#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot breakouts at R3/S3 with daily trend and volume spikes capture strong momentum moves in both bull and bear markets. The daily trend filter ensures we trade with the higher timeframe bias, while volume spikes confirm institutional participation. This combination reduces false breakouts and works across market regimes by requiring both trend alignment and volume confirmation.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    # Calculate R3, S3, R4, S4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    r4 = close_1d + (high_1d - low_1d) * 1.1
    s4 = close_1d - (high_1d - low_1d) * 1.1
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R3 with uptrend and volume confirmation
            if close[i] > r3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with downtrend and volume confirmation
            elif close[i] < s3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below S3 or trend reverses
            if close[i] < s3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above R3 or trend reverses
            if close[i] > r3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals