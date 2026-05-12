#!/usr/bin/env python3

# 4h_4H_RSI_34_SMA_50_Crossover_1dTrend_VolumeConfirm
# Hypothesis: On 4h timeframe, enter long when RSI(34) crosses above SMA(50) with volume >1.5x average and 1d EMA50 trending up; enter short when RSI(34) crosses below SMA(50) with volume >1.5x average and 1d EMA50 trending down.
# Uses daily trend filter to avoid counter-trend trades. Targets 15-25 trades/year to minimize fee drag and improve generalization across bull/bear markets.
# Focus on BTC/ETH as primary targets. Uses RSI crossover with trend confirmation for robustness.

name = "4h_4H_RSI_34_SMA_50_Crossover_1dTrend_VolumeConfirm"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate RSI(34) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Calculate SMA(50) on 4h close
    sma = pd.Series(close).rolling(window=50, min_periods=50).mean().values

    # Calculate RSI-SMA crossover signals
    rsi_above_sma = rsi > sma
    rsi_above_sma_prev = np.roll(rsi_above_sma, 1)
    rsi_above_sma_prev[0] = False
    crossover_up = rsi_above_sma & (~rsi_above_sma_prev)
    crossover_down = (~rsi_above_sma) & rsi_above_sma_prev

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI crosses above SMA + 1d uptrend + volume confirmation
            if (crossover_up[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below SMA + 1d downtrend + volume confirmation
            elif (crossover_down[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below SMA
            if crossover_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above SMA
            if crossover_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals