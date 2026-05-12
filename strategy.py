#!/usr/bin/env python3
"""
1d_Keltner_MeanReversion_Squeeze
Hypothesis: In low volatility (Keltner width < 20th percentile), price tends to revert to the 20-period EMA. Enter long when price < lower Keltner + RSI(14) < 30, short when price > upper Keltner + RSI(14) > 70. Exit when price crosses EMA(20). Works in both bull and bear markets by capturing mean-reversion in ranging conditions and avoiding trends via volatility filter. Uses 1d timeframe with 1w trend filter to avoid counter-trend trades.
Timeframe: 1d
"""

name = "1d_Keltner_MeanReversion_Squeeze"
timeframe = "1d"
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

    # Get weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Keltner Channel (20, 1.5)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_keltner = ema_20 + 1.5 * atr
    lower_keltner = ema_20 - 1.5 * atr
    keltner_width = upper_keltner - lower_keltner

    # Keltner width percentile (20-period lookback)
    width_series = pd.Series(keltner_width)
    width_percentile = width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 20 else np.nan,
        raw=False
    ).values

    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(width_percentile[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Low volatility squeeze: width < 20th percentile
            if width_percentile[i] < 20:
                # LONG: price < lower Keltner + RSI < 30 + above weekly EMA (weak uptrend bias)
                if (close[i] < lower_keltner[i] and 
                    rsi[i] < 30 and 
                    close[i] > ema_20_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: price > upper Keltner + RSI > 70 + below weekly EMA (weak downtrend bias)
                elif (close[i] > upper_keltner[i] and 
                      rsi[i] > 70 and 
                      close[i] < ema_20_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No trade in high volatility
        elif position == 1:
            # EXIT LONG: price crosses above EMA(20)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below EMA(20)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals