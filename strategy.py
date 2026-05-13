#!/usr/bin/env python3
# 4h_RSI_Volume_Trend_Filter
# Hypothesis: RSI mean reversion on 4h timeframe filtered by 1d trend and volume spikes.
# Long when RSI < 30 (oversold) with 1d uptrend and volume spike.
# Short when RSI > 70 (overbought) with 1d downtrend and volume spike.
# Exit when RSI returns to neutral range (40-60) or trend reverses.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Volume spike confirms institutional interest, reducing false signals.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_RSI_Volume_Trend_Filter"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2.0 * 10-period average (~1.6 days)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > 2.0 * vol_ma_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) + 1d uptrend + volume spike
            if rsi[i] < 30 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + 1d downtrend + volume spike
            elif rsi[i] > 70 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 40 (recovery) or trend reversal
            if rsi[i] > 40 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 60 (recovery) or trend reversal
            if rsi[i] < 60 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals