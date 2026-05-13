#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Trade Camarilla pivot breakouts on 4h with 1d EMA trend filter and volume confirmation.
# In bull markets, go long on breakout above R3 when price > 1d EMA34 with volume spike.
# In bear markets, go short on breakdown below S3 when price < 1d EMA34 with volume spike.
# Uses Camarilla levels from prior 1d candle (no look-ahead) and filters by 1d EMA34 trend.
# Volume > 1.5x 20-period average confirms breakout strength.
# Target: 80-160 total trades over 4 years = 20-40/year.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate Camarilla levels from prior 1d candle
    # H, L, C from previous day (already completed due to align_htf_to_ltf delay)
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We only need R3 and S3
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    
    # Align Camarilla levels to 4h (wait for prior 1d candle to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 + price > 1d EMA34 (bullish trend) + volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 + price < 1d EMA34 (bearish trend) + volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below S3 or price < 1d EMA34
            if (close[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above R3 or price > 1d EMA34
            if (close[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals