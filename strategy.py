#!/usr/bin/env python3
"""
6h_KeltnerChannel_RSI_Filter
Hypothesis: On 6h timeframe, price reversion to the mean occurs when price touches the Keltner Channel (KC) bands with RSI showing exhaustion. 
We use 20-period EMA as KC middle, ATR(10) for bands, and RSI(14) to confirm overbought/oversold conditions. 
Trend filter from 12h EMA50 ensures we only trade in direction of higher timeframe trend to avoid counter-trend whipsaws. 
Volume confirmation (>1.5x 20-period average) increases reliability of reversals. 
Designed for low turnover (12-37 trades/year) to minimize fee drag in both bull and bear markets.
"""

name = "6h_KeltnerChannel_RSI_Filter"
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

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate Keltner Channel components on 6h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema20 + 2 * atr10
    kc_lower = ema20 - 2 * atr10

    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar
        ema50 = ema50_12h_aligned[i]
        kc_up = kc_upper[i]
        kc_low = kc_lower[i]
        rsi_val = rsi[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(kc_up) or np.isnan(kc_low) or 
            np.isnan(rsi_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at or below KC lower + RSI oversold (<30) + price above 12h EMA50 + volume surge
            if (close[i] <= kc_low and 
                rsi_val < 30 and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above KC upper + RSI overbought (>70) + price below 12h EMA50 + volume surge
            elif (close[i] >= kc_up and 
                  rsi_val > 70 and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back above KC middle or RSI > 50
            if (close[i] >= ema20 or rsi_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back below KC middle or RSI < 50
            if (close[i] <= ema20 or rsi_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals