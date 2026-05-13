#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation, filtered by Choppiness Index to avoid range-bound markets.
# Uses Camarilla pivot levels from 1d for structure, 1d EMA34 for trend, volume spike for confirmation, and Choppiness Index > 61.8 to avoid false breakouts in chop.
# Designed to work in both bull and bear markets by filtering counter-trend trades and avoiding whipsaws in low-volatility regimes.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla pivot levels (R1, S1) from 1d data
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate Choppiness Index (14) on 12h data
    atr14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((atr14 * 14) / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # Neutral value when undefined

    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Camarilla R1 with volume spike, uptrend, and not choppy
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 with volume spike, downtrend, and not choppy
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals