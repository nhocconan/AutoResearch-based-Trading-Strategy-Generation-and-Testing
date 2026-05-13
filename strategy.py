#/usr/bin/env python3
# 4h_Keltner_RSI_MeanReversion_Trend
# Hypothesis: Keltner Channel combined with RSI mean reversion and 1d EMA trend filter works in both bull and bear.
# In bull: buy pullbacks to lower Keltner band when RSI < 40 and price > 1d EMA200.
# In bear: sell rallies to upper Keltner band when RSI > 60 and price < 1d EMA200.
# Uses 20-period ATR(1.5) for Keltner width. Target: 20-40 trades/year.

name = "4h_Keltner_RSI_MeanReversion_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Keltner Channel: EMA(20) ± ATR(20) * 1.5
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + atr * 1.5
    lower_keltner = ema20 - atr * 1.5

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price < lower Keltner + RSI < 40 (oversold) + price > 1d EMA200 (uptrend filter)
            if (close[i] < lower_keltner[i] and 
                rsi[i] < 40 and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price > upper Keltner + RSI > 60 (overbought) + price < 1d EMA200 (downtrend filter)
            elif (close[i] > upper_keltner[i] and 
                  rsi[i] > 60 and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price > EMA20 (mean reversion) or RSI > 50
            if (close[i] > ema20[i] or rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price < EMA20 (mean reversion) or RSI < 50
            if (close[i] < ema20[i] or rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals