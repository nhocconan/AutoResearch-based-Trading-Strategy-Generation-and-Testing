#!/usr/bin/env python3
# 4h_True_Strength_Index_Momentum
# Hypothesis: Use True Strength Index (TSI) to capture momentum with reduced whipsaw, confirmed by 1d trend (EMA34) and volume spikes (>1.8x 20-period average). Enter long when TSI crosses above its signal line and price > 1d EMA34 with volume spike; short when TSI crosses below signal line and price < 1d EMA34 with volume spike. Exit on opposite TSI crossover. Targets 20-40 trades/year to minimize fee decay while capturing momentum in both bull and bear markets via trend filter.

name = "4h_True_Strength_Index_Momentum"
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
    open_ = prices['open'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # True Strength Index (TSI) calculation
    # Step 1: Price change and absolute price change
    price_change = close - np.roll(close, 1)
    abs_price_change = np.abs(price_change)
    price_change[0] = 0  # First value has no previous close
    abs_price_change[0] = 0

    # Step 2: Double smoothed EMA of price change and abs price change
    # First smoothing (25-period EMA)
    pc_smoothed1 = pd.Series(price_change).ewm(span=25, adjust=False, min_periods=25).mean().values
    apc_smoothed1 = pd.Series(abs_price_change).ewm(span=25, adjust=False, min_periods=25).mean().values
    # Second smoothing (13-period EMA)
    pc_smoothed2 = pd.Series(pc_smoothed1).ewm(span=13, adjust=False, min_periods=13).mean().values
    apc_smoothed2 = pd.Series(apc_smoothed1).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Avoid division by zero
    apc_smoothed2_safe = np.where(apc_smoothed2 == 0, 1e-10, apc_smoothed2)
    tsi = (pc_smoothed2 / apc_smoothed2_safe) * 100

    # Signal line (7-period EMA of TSI)
    tsi_signal = pd.Series(tsi).ewm(span=7, adjust=False, min_periods=7).mean().values

    # TSI crossover signals
    tsi_cross_up = (tsi > tsi_signal) & (np.roll(tsi, 1) <= np.roll(tsi_signal, 1))
    tsi_cross_down = (tsi < tsi_signal) & (np.roll(tsi, 1) >= np.roll(tsi_signal, 1))

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(25, n):
        # Skip if any required value is NaN
        if (np.isnan(tsi[i]) or np.isnan(tsi_signal[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TSI crosses above signal line + price > 1d EMA34 + volume spike
            if (tsi_cross_up[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: TSI crosses below signal line + price < 1d EMA34 + volume spike
            elif (tsi_cross_down[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TSI crosses below signal line
            if tsi_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TSI crosses above signal line
            if tsi_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals