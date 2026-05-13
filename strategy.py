#!/usr/bin/env python3
# 4h_RSI_20_80_Trend_Volume_Spike
# Hypothesis: Use RSI extremes (20/80) for mean-reversion entries, confirmed by 1d EMA trend and volume spikes.
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies) by filtering with trend.
# Volume spike ensures institutional participation. Low-frequency signals to avoid fee drag.

name = "4h_RSI_20_80_Trend_Volume_Spike"
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

    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 20 (oversold) with volume spike and uptrend
            if rsi[i] < 20 and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 80 (overbought) with volume spike and downtrend
            elif rsi[i] > 80 and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 60 (overbought) or trend turns down
            if rsi[i] > 60 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 40 (oversold) or trend turns up
            if rsi[i] < 40 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals