# 1d_RSI200_Trend_Volume_Spike
# Hypothesis: Daily RSI(2) extreme reversals with 200-day EMA trend filter and volume spike.
# Works in bull/bear: RSI2 captures short-term exhaustion; EMA200 filters direction; volume confirms strength.
# Target: 10-25 trades/year to minimize fee drag.

name = "1d_RSI200_Trend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Daily EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Daily RSI(2) for short-term extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi2 = 100 - (100 / (1 + rs))
    rsi2 = rsi2.values

    # Volume spike: 2x 20-day average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        if np.isnan(ema200[i]) or np.isnan(rsi2[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI2 < 10 (oversold) + price > EMA200 (uptrend) + volume spike
            if (rsi2[i] < 10 and
                close[i] > ema200[i] and
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI2 > 90 (overbought) + price < EMA200 (downtrend) + volume spike
            elif (rsi2[i] > 90 and
                  close[i] < ema200[i] and
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI2 > 70 (overbought) or price < EMA200 (trend break)
            if (rsi2[i] > 70 or
                close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI2 < 30 (oversold) or price > EMA200 (trend break)
            if (rsi2[i] < 30 or
                close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3