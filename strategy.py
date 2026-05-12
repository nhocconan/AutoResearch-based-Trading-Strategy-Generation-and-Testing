#!/usr/bin/env python3
# 4h_4H_RSI_34_SMA_50_Crossover_1dTrend_VolumeConfirm
# Hypothesis: On 4h timeframe, buy when RSI(34) crosses above SMA(50) with volume > 1.5x average and 1d EMA50 trending up; sell when RSI crosses below SMA with volume confirmation and 1d EMA50 trending down. Uses 1d trend filter to avoid counter-trend trades and volume confirmation to ensure breakout strength. Targets 20-40 trades per year to minimize fee drift. Works in bull via momentum and in bear via short signals during downtrends.

name = "4h_4H_RSI_34_SMA_50_Crossover_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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

    # Calculate RSI(34)
    def rsi(close, period=34):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        return 100 - (100 / (1 + rs))

    rsi_34 = rsi(close, 34)

    # Calculate SMA(50)
    sma_50 = np.convolve(close, np.ones(50)/50, mode='same')
    # Adjust for edges: use expanding mean for first 50 periods
    for i in range(50):
        sma_50[i] = np.mean(close[:i+1])

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    for i in range(20):
        vol_avg_20[i] = np.mean(volume[:i+1])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_34[i]) or np.isnan(sma_50[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI crosses above SMA + 1d uptrend + volume spike
            if (rsi_34[i] > sma_50[i] and rsi_34[i-1] <= sma_50[i-1] and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below SMA + 1d downtrend + volume spike
            elif (rsi_34[i] < sma_50[i] and rsi_34[i-1] >= sma_50[i-1] and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below SMA OR trend turns down
            if rsi_34[i] < sma_50[i] and rsi_34[i-1] >= sma_50[i-1] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above SMA OR trend turns up
            if rsi_34[i] > sma_50[i] and rsi_34[i-1] <= sma_50[i-1] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals