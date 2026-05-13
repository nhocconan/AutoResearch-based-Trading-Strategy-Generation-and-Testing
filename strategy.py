#!/usr/bin/env python3
# 6h_TRIX_VolumeSpike_Regime
# Hypothesis: Use TRIX (triple exponential average momentum) to detect momentum exhaustion and reversals.
# Combine with volume spikes for confirmation and a regime filter based on ADX to avoid whipsaws.
# In trending markets (ADX > 25), follow TRIX crossovers; in ranging markets (ADX < 20), fade extreme TRIX values.
# This adapts to both bull and bear markets by using momentum direction and regime context.
# Volume spikes ensure participation, reducing false signals.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_TRIX_VolumeSpike_Regime"
timeframe = "6h"
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

    # TRIX calculation: triple EMA of log(close), then ROC
    # Using period=12 as standard
    log_close = np.log(close)
    ema1 = pd.Series(log_close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (pd.Series(ema3).diff(1) / pd.Series(ema3).shift(1))
    trix = trix.fillna(0).values

    # ADX for regime filtering (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr_period = 14
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(trix[i]) or np.isnan(adx[i]) or np.isnan(vol_avg_20[i]) or 
            atr[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Regime-based entry logic
            if adx[i] > 25:  # Trending regime
                # Long: TRIX crosses above zero with volume spike
                if trix[i] > 0 and trix[i-1] <= 0 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero with volume spike
                elif trix[i] < 0 and trix[i-1] >= 0 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 25)
                # Fade extreme TRIX values: long when TRIX very negative, short when very positive
                if trix[i] < -0.5 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = 0.25
                    position = 1
                elif trix[i] > 0.5 and volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or volume drops
            if trix[i] < 0 or volume[i] < vol_avg_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or volume drops
            if trix[i] > 0 or volume[i] < vol_avg_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals