#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use weekly (1w) EMA200 for trend filter and 1d Camarilla R1/S1 levels for breakout entries with volume confirmation.
# Long when price breaks above R1 in uptrend (price > weekly EMA200) with volume spike, short when price breaks below S1 in downtrend (price < weekly EMA200) with volume spike.
# Exit when price returns to 1d pivot point (PP). Designed for 20-50 trades/year to avoid overtrading.

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels: R1, S1, and PP (pivot point)
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp_1d = typical_price.values
    hl_range = df_1d['high'] - df_1d['low']
    r1_1d = df_1d['close'].values + hl_range.values * 1.1 / 12
    s1_1d = df_1d['close'].values - hl_range.values * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above weekly EMA200 (uptrend) + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below weekly EMA200 (downtrend) + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals