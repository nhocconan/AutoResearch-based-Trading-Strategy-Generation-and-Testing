#!/usr/bin/env python3
"""
1h_Pullback_Trend_Follower
Hypothesis: Use 4h EMA trend filter and 1d RSI filter to identify primary trend, then enter on 1h pullbacks with volume confirmation. Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends. Uses volume spike to confirm institutional interest during pullbacks, reducing false signals in chop. Target: 20-40 trades/year.
"""

name = "1h_Pullback_Trend_Follower"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for EMA trend (primary timeframe direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Get 1d data for RSI filter (avoid overextended conditions)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        ema50_val = ema50_4h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(ema50_val) or np.isnan(rsi_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend + 1d RSI not overbought + pullback to EMA + volume confirmation
            if ema50_val > close[i] and close[i] > ema50_val * 0.98 and rsi_val < 70 and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + 1d RSI not oversold + pullback to EMA + volume confirmation
            elif ema50_val < close[i] and close[i] < ema50_val * 1.02 and rsi_val > 30 and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h downtrend or RSI overbought
            if ema50_val < close[i] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h uptrend or RSI oversold
            if ema50_val > close[i] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals