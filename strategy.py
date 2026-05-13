#!/usr/bin/env python3
# 4h_12h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Use 12h Camarilla pivot points (R1/S1) for breakout entries with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h R1 in uptrend (price > 12h EMA50) with volume spike, short when price breaks below 12h S1 in downtrend (price < 12h EMA50) with volume spike.
# Exit when price returns to 12h pivot point (PP) or trend changes.
# 12h pivots provide stronger support/resistance than daily, reducing false breakouts in volatile markets.
# Designed for 4h timeframe with HTF 12h to avoid overtrading and capture multi-day trends.

name = "4h_12h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot points: PP, R1, S1
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    r1_12h = df_12h['close'] + (df_12h['high'] - df_12h['low']) * 1.1 / 12
    s1_12h = df_12h['close'] - (df_12h['high'] - df_12h['low']) * 1.1 / 12
    
    # Align 12h Camarilla levels to 4h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h.values)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h.values)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h.values)

    # Get 12h data for EMA trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(pp_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above 12h EMA50 (uptrend) + volume spike
            if (close[i] > r1_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below 12h EMA50 (downtrend) + volume spike
            elif (close[i] < s1_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_12h_aligned[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_12h_aligned[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals