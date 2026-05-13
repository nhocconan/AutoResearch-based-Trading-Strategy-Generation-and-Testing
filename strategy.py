#!/usr/bin/env python3
# 12h_WilliamsFractal_Breakout_1wTrend_Volume
# Hypothesis: Williams Fractal breakouts with 1-week EMA50 trend filter and volume confirmation capture momentum while minimizing trades. Works in bull markets via breakouts above bearish fractals and in bear markets via breakdowns below bullish fractals. Uses 1-week EMA50 to filter trend direction and volume spike for confirmation, reducing false signals. Target: 12-37 trades per year per symbol to minimize fee drag.

name = "12h_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # ATR for stoploss context (not used in signal)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Get 1w data for Williams Fractals and EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Fractals (requires 2-bar confirmation after center)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Align with 2-bar additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )

    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: >2.0x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above bearish fractal + 1w EMA50 uptrend + volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below bullish fractal + 1w EMA50 downtrend + volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below bullish fractal or volatility drop
            if close[i] < bullish_fractal_aligned[i] or volume[i] < vol_avg_30[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above bearish fractal or volatility drop
            if close[i] > bearish_fractal_aligned[i] or volume[i] < vol_avg_30[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals