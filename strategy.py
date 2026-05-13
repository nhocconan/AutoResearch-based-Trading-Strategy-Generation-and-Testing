#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Use daily Camarilla pivot points (R1/S1) for breakout entries with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R1 in uptrend with volume spike, short when price breaks below S1 in downtrend with volume spike.
# Exit when price returns to daily pivot point (PP) or trend changes.
# Uses 4h timeframe to limit trade frequency and improve generalization in both bull and bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot points: PP, R1, S1
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    hlc = df_1d['high'] + df_1d['low'] + df_1d['close']
    pp_1d = hlc / 3
    range_hl = df_1d['high'] - df_1d['low']
    r1_1d = df_1d['close'] + range_hl * 1.1 / 12
    s1_1d = df_1d['close'] - range_hl * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)

    # Get daily data for EMA trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above 1d EMA34 (uptrend) + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below 1d EMA34 (downtrend) + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA34)
            if (close[i] <= pp_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA34)
            if (close[i] >= pp_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals