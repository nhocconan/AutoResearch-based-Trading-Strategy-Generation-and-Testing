#!/usr/bin/env python3
# 6h_Engulfing_BullBear_Momentum
# Hypothesis: Use bullish/bearish engulfing candles on 6h with 1d EMA trend filter and volume confirmation.
# Engulfing candles signal strong momentum reversals; EMA filter ensures trades align with higher-timeframe trend.
# Works in bull (follows bullish engulfing in bullish 1d trend) and bear (avoids bullish engulfing in bearish 1d trend, takes bearish engulfing).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Engulfing_BullBear_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Detect bullish engulfing: current bullish candle engulfs previous bearish candle
        bullish_engulfing = (close[i] > open_[i]) and (open_[i] < close[i-1]) and (close[i] > open_[i-1]) and (open_[i-1] > close[i-1])
        # Detect bearish engulfing: current bearish candle engulfs previous bullish candle
        bearish_engulfing = (close[i] < open_[i]) and (open_[i] > close[i-1]) and (close[i] < open_[i-1]) and (open_[i-1] < close[i-1])

        if position == 0:
            # LONG: bullish engulfing + price above 1d EMA (bullish trend) + volume spike
            if (bullish_engulfing and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish engulfing + price below 1d EMA (bearish trend) + volume spike
            elif (bearish_engulfing and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish engulfing or price below 1d EMA
            if (bearish_engulfing or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish engulfing or price above 1d EMA
            if (bullish_engulfing or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals