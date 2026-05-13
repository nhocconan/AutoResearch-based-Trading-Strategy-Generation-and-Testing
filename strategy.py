#!/usr/bin/env python3
# 1h_RSI_MeanReversion_4hTrend_Filter
# Hypothesis: Mean reversion on 1h RSI extremes filtered by 4h trend direction.
# Long when RSI < 30 (oversold) and price > 4h EMA50 (uptrend).
# Short when RSI > 70 (overbought) and price < 4h EMA50 (downtrend).
# Exit when RSI returns to neutral (40-60) or trend reverses.
# Uses 4h trend to avoid counter-trend whipsaws, targeting 15-35 trades/year per symbol.
# Works in bull (trend-following pullbacks) and bear (counter-trend bounces) markets.

name = "1h_RSI_MeanReversion_4hTrend_Filter"
timeframe = "1h"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # 1h RSI(14) - Wilder's smoothing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # start after RSI warmup
        # Skip if any required value is NaN
        if np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) in 4h uptrend
            if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI overbought (>70) in 4h downtrend
            elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40) or trend turns down
            if rsi[i] >= 40 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60) or trend turns up
            if rsi[i] <= 60 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals