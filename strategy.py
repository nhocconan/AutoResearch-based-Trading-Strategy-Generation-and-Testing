#!/usr/bin/env python3
"""
4h_RSI_Pullback_to_EMA_with_Volume_Spike
Hypothesis: In strong trends (ADX > 25), price pulls back to the 21-period EMA offering high-probability entries. 
RSI < 30 (oversold) for longs, RSI > 70 (overbought) for shorts, combined with volume spikes (>1.5x 20-bar average) 
to confirm momentum resumption. Trend filter uses 1d EMA50 to ensure alignment with higher timeframe direction. 
Works in both bull and bear markets by trading pullbacks within the dominant trend.
"""

name = "4h_RSI_Pullback_to_EMA_with_Volume_Spike"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # EMA21 for pullback target
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(rsi[i]) or np.isnan(ema21[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold), price near EMA21, volume spike, ADX > 25, 1d uptrend
            if (rsi[i] < 30 and 
                abs(close[i] - ema21[i]) / ema21[i] < 0.02 and  # within 2% of EMA21
                volume[i] > vol_avg_20[i] * 1.5 and
                adx[i] > 25 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought), price near EMA21, volume spike, ADX > 25, 1d downtrend
            elif (rsi[i] > 70 and 
                  abs(close[i] - ema21[i]) / ema21[i] < 0.02 and  # within 2% of EMA21
                  volume[i] > vol_avg_20[i] * 1.5 and
                  adx[i] > 25 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (momentum fading) or price breaks below EMA21
            if rsi[i] > 50 or close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (momentum fading) or price breaks above EMA21
            if rsi[i] < 50 or close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals