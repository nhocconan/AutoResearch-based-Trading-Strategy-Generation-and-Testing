#!/usr/bin/env python3
# 4h_RSI_40_60_Momentum_1dTrend_Volume
# Hypothesis: Use RSI(14) momentum (40-60 range) with 1d EMA50 trend filter and volume confirmation.
# RSI between 40 and 60 indicates balanced momentum without overextension. Enter long when RSI > 50 in bullish 1d trend,
# short when RSI < 50 in bearish 1d trend, both with volume spike. Avoids overextended RSI extremes.
# Works in bull (follows momentum with bullish 1d trend) and bear (avoids bullish momentum in bearish 1d trend).
# Target: 60-120 total trades over 4 years = 15-30/year.

name = "4h_RSI_40_60_Momentum_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50 (bullish momentum) + price above 1d EMA (bullish trend) + volume spike
            if (rsi_values[i] > 50 and 
                close[i] > ema_50_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 (bearish momentum) + price below 1d EMA (bearish trend) + volume spike
            elif (rsi_values[i] < 50 and 
                  close[i] < ema_50_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 (oversold) or price below 1d EMA
            if (rsi_values[i] < 40 or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 60 (overbought) or price above 1d EMA
            if (rsi_values[i] > 60 or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals