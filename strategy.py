#!/usr/bin/env python3
"""
4h_Inside_Bar_Failure_Mean_Reversion
Hypothesis: Inside bars (range contraction) followed by breakout failures indicate rejection and mean reversion opportunities.
In bull markets: short after failed upside breakout from inside bar.
In bear markets: long after failed downside breakout from inside bar.
Uses 1d ATR for volatility filter and volume confirmation to avoid low-quality signals.
Designed for 20-50 trades/year on 4h timeframe with clear entry/exit rules.
"""

name = "4h_Inside_Bar_Failure_Mean_Reversion"
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

    # Get daily data (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)

    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values

    # Calculate 14-day ATR for volatility filter
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr14_daily)

    # Volume confirmation: volume > 1.3x 20-period average (on 4h timeframe)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        atr_val = atr14_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(atr_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: avoid low volatility environments
        if atr_val < 0.01 * close[i]:  # Skip if ATR too low relative to price
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Detect inside bar: current range within previous bar's range
        is_inside_bar = (high[i] <= high[i-1]) and (low[i] >= low[i-1])

        if position == 0:
            # Only trade after an inside bar
            if is_inside_bar:
                # LONG: Failed downside breakout (close above inside bar high)
                if close[i] > high[i-1] and volume[i] > vol_avg_val * 1.3:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Failed upside breakout (close below inside bar low)
                elif close[i] < low[i-1] and volume[i] > vol_avg_val * 1.3:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below inside bar low or opposite failed breakout
            if close[i] < low[i-1] or (high[i] <= high[i-1] and low[i] >= low[i-1] and close[i] < high[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above inside bar high or opposite failed breakout
            if close[i] > high[i-1] or (high[i] <= high[i-1] and low[i] >= low[i-1] and close[i] > low[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals