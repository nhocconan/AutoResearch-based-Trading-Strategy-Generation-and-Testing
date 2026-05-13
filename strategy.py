#!/usr/bin/env python3
"""
4h_Keltner_RSI_Trend_Filter_With_Volume_Confirmation
Hypothesis: In trending markets, price tends to pull back to the 20-period EMA (Keltner middle band)
before continuing in the trend direction. We use:
- 4h EMA200 trend filter to align with higher timeframe momentum
- Keltner Channel (20, 2.0) for dynamic support/resistance
- RSI(14) to avoid overextended entries (long when RSI<60, short when RSI>40)
- Volume spike (2x 20-period average) to confirm institutional participation
This combines trend-following with pullback entries and volume confirmation for high-probability
trades in both bull and bear markets. Targets low-frequency, high-quality setups.
"""

name = "4h_Keltner_RSI_Trend_Filter_With_Volume_Confirmation"
timeframe = "4h"
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

    # 4h EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Keltner Channel: 20-period EMA, 2.0 * ATR(10) for bands
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema200[i]) or 
            np.isnan(ema20[i]) or
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price near Keltner lower band + RSI not overbought + volume spike
            if close[i] > ema200[i] and close[i] <= kc_lower[i] * 1.005 and rsi[i] < 60 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price near Keltner upper band + RSI not oversold + volume spike
            elif close[i] < ema200[i] and close[i] >= kc_upper[i] * 0.995 and rsi[i] > 40 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Keltner upper band or trend turns bearish
            if close[i] >= kc_upper[i] * 0.995 or close[i] < ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Keltner lower band or trend turns bullish
            if close[i] <= kc_lower[i] * 1.005 or close[i] > ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals