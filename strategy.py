#!/usr/bin/env python3
"""
6h_Keltner_Channel_Trend_Volume
Hypothesis: On 6h timeframe, price breaking above/below the Keltner Channel with
20-period ATR(10) and EMA20 trend filter, combined with volume > 2x 20-period
average, captures momentum moves. Keltner Channels adapt to volatility better
than fixed bands, reducing whipsaws in sideways markets. Volume confirmation
ensures breakouts have conviction. Designed for 12-37 trades/year.
"""

name = "6h_Keltner_Channel_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Keltner Channel parameters
    atr_period = 10
    ema_period = 20
    keltner_mult = 2.0
    vol_mult = 2.0

    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values

    # Calculate EMA for middle line
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values

    # Calculate Keltner Bands
    upper = ema + keltner_mult * atr
    lower = ema - keltner_mult * atr

    # Volume confirmation: 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(ema_period, n):
        # Skip if any required data is NaN
        if (np.isnan(ema[i]) or np.isnan(atr[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above upper Keltner band + price above EMA + volume surge
            if (close[i] > upper[i] and 
                close[i] > ema[i] and 
                volume[i] > vol_avg_20[i] * vol_mult):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below lower Keltner band + price below EMA + volume surge
            elif (close[i] < lower[i] and 
                  close[i] < ema[i] and 
                  volume[i] > vol_avg_20[i] * vol_mult):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA (trend change)
            if close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA (trend change)
            if close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals