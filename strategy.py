#!/usr/bin/env python3
# 6h_ThreeBarReversal_PivotBreakout_1dTrend_Volume
# Hypothesis: Three-bar reversal patterns at daily pivot levels with volume confirmation.
# In trending markets (1d EMA34), wait for 3-bar reversal at S1/R1 levels, then enter on breakout.
# Long: Bullish 3-bar reversal at S1, then break above R1 with volume spike.
# Short: Bearish 3-bar reversal at R1, then break below S1 with volume spike.
# Exit on opposite pivot touch or 3-bar reversal in opposite direction.
# Combines price action (3-bar reversal), structure (pivots), and trend/volume filters.
# Target: 20-30 trades/year on 6h to minimize fee decay while capturing high-probability swings.

name = "6h_ThreeBarReversal_PivotBreakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points (standard: PP, R1, S1)
    # Based on previous day's OHLC
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d

    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    # Three-bar reversal detection
    # Bullish: low < previous low AND low < next low (hammer-like)
    # Bearish: high > previous high AND high > next high (shooting star-like)
    bullish_reversal = np.zeros(n, dtype=bool)
    bearish_reversal = np.zeros(n, dtype=bool)

    for i in range(1, n-1):
        if low[i] < low[i-1] and low[i] < low[i+1]:
            bullish_reversal[i] = True
        if high[i] > high[i-1] and high[i] > high[i+1]:
            bearish_reversal[i] = True

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish 3-bar reversal at S1, then break above R1 with volume
            if (bullish_reversal[i-1] and  # reversal confirmed on prior bar
                low[i-1] <= s1_aligned[i-1] * 1.005 and  # near S1 (within 0.5%)
                close[i] > r1_aligned[i] and  # break above R1
                close[i] > ema34_1d_aligned[i] and  # above trend
                volume[i] > vol_avg_30[i] * 2.0):  # volume spike
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 3-bar reversal at R1, then break below S1 with volume
            elif (bearish_reversal[i-1] and  # reversal confirmed on prior bar
                  high[i-1] >= r1_aligned[i-1] * 0.995 and  # near R1 (within 0.5%)
                  close[i] < s1_aligned[i] and  # break below S1
                  close[i] < ema34_1d_aligned[i] and  # below trend
                  volume[i] > vol_avg_30[i] * 2.0):  # volume spike
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches S1 (mean reversion) OR bearish reversal at R1
            if (close[i] <= s1_aligned[i] or 
                (bearish_reversal[i] and high[i] >= r1_aligned[i] * 0.995)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches R1 (mean reversion) OR bullish reversal at S1
            if (close[i] >= r1_aligned[i] or 
                (bullish_reversal[i] and low[i] <= s1_aligned[i] * 1.005)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals