#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike captures momentum in both bull and bear regimes.
# Uses price rejection at key intraday pivot levels for mean-reversion exits.
# Entry: Long when close > R1 + 12h EMA50 uptrend + volume > 1.5x 20-period average; Short when close < S1 + 12h EMA50 downtrend + volume spike.
# Exit: Mean reversion to Camarilla Pivot Point (PP) to avoid overstaying in extended moves.
# Target: 25-35 trades/year on 4h to stay within optimal range while capturing significant moves.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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

    # Calculate previous day's high, low, close for Camarilla levels
    # Shift by 1 to use previous day's data only
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else 0
    prev_low[0] = prev_low[1] if n > 1 else 0
    prev_close[0] = prev_close[1] if n > 1 else 0

    # Camarilla levels: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0

    # 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(PP[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 12h EMA50 uptrend + volume spike
            if (close[i] > R1[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + 12h EMA50 downtrend + volume spike
            elif (close[i] < S1[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to Camarilla Pivot Point (PP)
            if close[i] < PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to Camarilla Pivot Point (PP)
            if close[i] > PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals