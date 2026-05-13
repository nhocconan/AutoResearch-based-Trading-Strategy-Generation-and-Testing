#!/usr/bin/env python3
# 4h_ThreeLevelBreakout_Pattern
# Hypothesis: Combines multiple breakout levels (Donchian, Keltner, ATR-based) with volume confirmation and trend filter.
# Uses three confirmation layers to reduce false signals and maintain low trade frequency.
# Designed to work in both bull and bear markets by requiring trend alignment and volume spikes.
# Target: 20-40 trades per year per symbol with disciplined risk management.

name = "4h_ThreeLevelBreakout_Pattern"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Calculate ATR for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Level 1: Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Level 2: Keltner Channel (20-period EMA + 2*ATR)
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kelt_upper = ema_mid + 2 * atr
    kelt_lower = ema_mid - 2 * atr

    # Level 3: ATR-based breakout levels
    atr_upper = donch_high + 0.5 * atr
    atr_lower = donch_low - 0.5 * atr

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Trend filter: 50-period EMA on price
    ema_trend = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(kelt_upper[i]) or np.isnan(kelt_lower[i]) or
            np.isnan(atr_upper[i]) or np.isnan(atr_lower[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above all three levels with volume spike and uptrend
            if (close[i] > donch_high[i] and 
                close[i] > kelt_upper[i] and 
                close[i] > atr_upper[i] and
                volume_spike[i] and 
                close[i] > ema_trend[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below all three levels with volume spike and downtrend
            elif (close[i] < donch_low[i] and 
                  close[i] < kelt_lower[i] and 
                  close[i] < atr_lower[i] and
                  volume_spike[i] and 
                  close[i] < ema_trend[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below the middle Keltner level
            if close[i] < ema_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above the middle Keltner level
            if close[i] > ema_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals