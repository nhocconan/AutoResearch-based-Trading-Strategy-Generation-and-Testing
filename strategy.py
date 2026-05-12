#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_TrendFilter_Volume
# Hypothesis: Use TRIX (15) zero cross as momentum signal with 1d EMA50 trend filter and volume confirmation.
# Long when TRIX crosses above zero with rising volume and price above daily EMA50.
# Short when TRIX crosses below zero with rising volume and price below daily EMA50.
# Exit when TRIX returns to zero or trend reverses.
# Works in bull markets (captures momentum) and bear markets (shorts during downtrends).
# Targets 20-30 trades/year by requiring zero cross, volume spike, and trend alignment.

name = "4h_TRIX_ZeroCross_TrendFilter_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate TRIX (15,9,9) - triple EMA of 1-period % change
    # ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.diff(close, prepend=close[0]) / np.where(close[:-1] == 0, 1e-10, close[:-1])
    roc = np.append(roc[1:], 0)  # align length
    
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA3 of EMA2 = TRIX
    trix = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values * 100  # scale to percentage
    
    # TRIX zero cross signals
    trix_above = trix > 0
    trix_below = trix < 0
    trix_cross_up = (trix > 0) & (np.roll(trix, 1) <= 0)  # crossed above zero
    trix_cross_down = (trix < 0) & (np.roll(trix, 1) >= 0)  # crossed below zero
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x average of last 6 periods (1.5 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(trix_cross_up[i]) or np.isnan(trix_cross_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from daily EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: TRIX crosses above zero with volume and uptrend
            if trix_cross_up[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume and downtrend
            elif trix_cross_down[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX returns to zero or trend turns down
            if trix[i] <= 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX returns to zero or trend turns up
            if trix[i] >= 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals