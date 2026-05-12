#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dTrend_Volume
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
In trending markets (ADX > 25), follow KAMA direction with volume confirmation.
In ranging markets (ADX <= 25), fade KAMA extremes at Bollinger Bands with volume confirmation.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency (<50/year) to minimize fee drag.
"""

name = "4h_KAMA_Direction_1dTrend_Volume"
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

    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    # Pad ER and SC arrays
    er = np.concatenate([np.full(9, np.nan), er])
    sc = np.concatenate([np.full(9, np.nan), sc])

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # ADX for regime detection (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Pad ADX arrays
    plus_dm = np.concatenate([np.zeros(1), plus_dm])
    minus_dm = np.concatenate([np.zeros(1), minus_dm])
    tr = np.concatenate([np.zeros(1), tr])
    atr_14 = np.concatenate([np.full(13, np.nan), atr_14])
    plus_di = np.concatenate([np.full(13, np.nan), plus_di])
    minus_di = np.concatenate([np.full(13, np.nan), minus_di])
    dx = np.concatenate([np.full(27, np.nan), dx])
    adx = np.concatenate([np.full(27, np.nan), adx])

    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]) or np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # TRENDING MARKET (ADX > 25): follow KAMA direction
            if adx[i] > 25:
                if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    if close[i] > ema50_1d_aligned[i]:  # 1d uptrend filter
                        signals[i] = 0.25
                        position = 1
                elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    if close[i] < ema50_1d_aligned[i]:  # 1d downtrend filter
                        signals[i] = -0.25
                        position = -1
            # RANGING MARKET (ADX <= 25): fade KAMA extremes at Bollinger Bands
            else:
                if close[i] <= lower_bb[i] and close[i-1] > lower_bb[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb[i] and close[i-1] < upper_bb[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: KAMA cross down OR 1d trend fails OR Bollinger Band touch (in range)
            if adx[i] > 25:
                if close[i] < kama[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                if close[i] >= sma20[i] or volume[i] <= vol_avg_20[i] * 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA cross up OR 1d trend fails OR Bollinger Band touch (in range)
            if adx[i] > 25:
                if close[i] > kama[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                if close[i] <= sma20[i] or volume[i] <= vol_avg_20[i] * 1.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals