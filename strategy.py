#!/usr/bin/env python3
# 6h_Keltner_Breakout_1dTrend_Volume
# Hypothesis: Use Keltner Channel (ATR-based) breakout for entries with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above upper Keltner band in uptrend with volume spike, short when price breaks below lower band in downtrend with volume spike.
# Exit when price returns to EMA200 or trend changes. Keltner channels adapt to volatility, reducing false breakouts in ranging markets.
# Designed for moderate trade frequency (50-150 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "6h_Keltner_Breakout_1dTrend_Volume"
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

    # Get daily data for EMA200 trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Calculate ATR(20) on daily timeframe for Keltner Channel
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20_1d = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)

    # Calculate Keltner Channel on 6h timeframe using daily ATR
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20_6h + (2 * atr_20_1d_aligned)
    lower_keltner = ema_20_6h - (2 * atr_20_1d_aligned)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > upper_keltner[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < lower_keltner[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to EMA200 or trend changes (price below EMA200)
            if (close[i] <= ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA200 or trend changes (price above EMA200)
            if (close[i] >= ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals