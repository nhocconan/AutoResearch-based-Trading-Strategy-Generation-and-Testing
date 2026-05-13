#!/usr/bin/env python3
# 6h_RangeReversal_Bollinger_RSI
# Hypothesis: In ranging markets (BTC/ETH 2025), price reverses at Bollinger Bands ±2σ with RSI extremes.
# Long when price touches lower BB, RSI<30, and volume spikes; short at upper BB, RSI>70, volume spike.
# Exit when price returns to middle BB (20 SMA). Uses 1d trend filter to avoid counter-trend trades.
# Designed for low turnover in ranging markets, works in both bull (buy dips) and bear (sell rallies).

name = "6h_RangeReversal_Bollinger_RSI"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev

    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: current > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(basis.iloc[i]) or np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]) or
            np.isnan(rsi.iloc[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price at lower BB, RSI oversold, volume spike, and 1d uptrend
            if close[i] <= lower.iloc[i] and rsi.iloc[i] < 30 and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price at upper BB, RSI overbought, volume spike, and 1d downtrend
            elif close[i] >= upper.iloc[i] and rsi.iloc[i] > 70 and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to middle BB
            if close[i] >= basis.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to middle BB
            if close[i] <= basis.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals