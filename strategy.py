#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout from daily Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
# Long when price breaks above R3 during 1d uptrend with volume spike; short when breaks below S3 in 1d downtrend.
# Exit on opposite Camarilla level (R1/S1) or trend reversal. Designed for 12h timeframe to reduce trade frequency.
# Works in bull (trend-following breakouts) and bear (mean-reversion bounces at S3/R3) markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    H_1d = df_1d['high'].values
    L_1d = df_1d['low'].values
    C_1d = df_1d['close'].values
    range_1d = H_1d - L_1d
    
    # Calculate levels for previous day (shifted by 1 to avoid look-ahead)
    R3_1d = C_1d + range_1d * 1.1 / 4
    S3_1d = C_1d - range_1d * 1.1 / 4
    R1_1d = C_1d + range_1d * 1.1 / 12
    S1_1d = C_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (available after 1d bar closes)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)

    # 1d EMA34 for trend direction
    ema_34_1d = pd.Series(C_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 in 1d uptrend with volume spike
            if close[i] > R3_1d_aligned[i]:
                if close[i] > ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S3 in 1d downtrend with volume spike
            elif close[i] < S3_1d_aligned[i]:
                if close[i] < ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below R1 or trend turns down
            if close[i] < R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above S1 or trend turns up
            if close[i] > S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals