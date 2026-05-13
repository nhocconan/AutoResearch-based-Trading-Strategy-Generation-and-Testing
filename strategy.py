#!/usr/bin/env python3
# 6h_RSI_Trend_Filter_with_Volume_Spike
# Hypothesis: RSI extremes on 6h timeframe (RSI<20 or >80) with trend filter (1d EMA100) and volume spike to confirm momentum.
# In bull markets: buy RSI<20 pullbacks in uptrend with volume; sell RSI>80 overextensions in downtrend with volume.
# In bear markets: sell RSI>80 rallies in downtrend with volume; buy RSI<20 bounces in uptrend with volume.
# Uses 1d EMA100 for trend to avoid counter-trend trades. Volume spike filters low-conviction moves.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "6h_RSI_Trend_Filter_with_Volume_Spike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA100
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate 6-period RSI on 6h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2.0 * 20-period average (~5 days worth at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 20 (oversold) + price above 1d EMA100 (uptrend) + volume spike
            if rsi[i] < 20 and close[i] > ema100_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 80 (overbought) + price below 1d EMA100 (downtrend) + volume spike
            elif rsi[i] > 80 and close[i] < ema100_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 (overbought) or trend reversal (price below EMA100)
            if rsi[i] > 70 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 (oversold) or trend reversal (price above EMA100)
            if rsi[i] < 30 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals