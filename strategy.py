#!/usr/bin/env python3
"""
1d_WV3_Filter_Signal
Hypothesis: WV3 (wave trend) oscillator combined with 1-week RSI trend filter and volume confirmation captures sustainable moves while avoiding whipsaws. Works in bull markets via momentum continuation and in bear via mean-reversion extremes filtered by higher timeframe trend.
"""

name = "1d_WV3_Filter_Signal"
timeframe = "1d"
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
    high = prices['high'].values
    low = prices['low'].values

    # Get 1w data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1w RSI14 for trend
    delta = pd.Series(close_1w).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)

    # WV3 (Wave Trend) calculation: EMA of (HL2 - SMA(HL2,10)), then double EMA
    hl2 = (high + low) / 2
    esa = pd.Series(hl2).ewm(span=10, adjust=False, min_periods=10).mean().values
    d = pd.Series(np.abs(hl2 - esa)).ewm(span=10, adjust=False, min_periods=10).mean().values
    ci = (hl2 - esa) / (0.015 * d)
    tci1 = pd.Series(ci).ewm(span=21, adjust=False, min_periods=21).mean().values
    wv3 = pd.Series(tci1).ewm(span=42, adjust=False, min_periods=42).mean().values

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(42, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(wv3[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: WV3 oversold (< -60) turning up + 1w RSI > 50 (bullish bias) + volume confirmation
            if wv3[i] < -60 and wv3[i] > wv3[i-1] and rsi_1w_aligned[i] > 50 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: WV3 overbought (> 60) turning down + 1w RSI < 50 (bearish bias) + volume confirmation
            elif wv3[i] > 60 and wv3[i] < wv3[i-1] and rsi_1w_aligned[i] < 50 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WV3 crosses above zero or 1w RSI drops below 40
            if wv3[i] > 0 or rsi_1w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WV3 crosses below zero or 1w RSI rises above 60
            if wv3[i] < 0 or rsi_1w_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals