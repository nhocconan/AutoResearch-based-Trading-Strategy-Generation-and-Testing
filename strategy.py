#!/usr/bin/env python3
# 12h_Keltner_Breakout_ATR_Volume
# Hypothesis: Use 12-hour Keltner Channel breakouts filtered by 1-day EMA trend and volume confirmation.
# Keltner Channels (EMA ± ATR multiplier) adapt to volatility, reducing false breakouts in low volatility.
# The 1d EMA50 filter ensures trades align with the daily trend, improving edge in both bull and bear markets.
# Volume confirmation adds conviction to breakout moves. Works in bull markets (follows upward breaks with bullish trend)
# and bear markets (avoids upward breaks in bearish trend, takes downward breaks).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Keltner_Breakout_ATR_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate ATR(14) on 12h for Keltner Channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Calculate EMA(20) on 12h for Keltner Channel midline
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Keltner Upper + price above 1d EMA50 (bullish trend) + volume spike
            if (close[i] > keltner_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Keltner Lower + price below 1d EMA50 (bearish trend) + volume spike
            elif (close[i] < keltner_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Keltner Lower or price below 1d EMA50
            if (close[i] < keltner_lower[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Keltner Upper or price above 1d EMA50
            if (close[i] > keltner_upper[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals