#!/usr/bin/env python3
# 6h_Williams_Fractal_Breakout_1dTrend_Volume
# Hypothesis: Williams Fractals from daily timeframe identify key support/resistance levels.
# Breakouts above bearish fractals (resistance) with volume confirmation and daily trend alignment go long.
# Breakdowns below bullish fractals (support) with volume confirmation and daily trend alignment go short.
# Works in both bull and bear markets by trading breakouts of significant daily levels with trend filter.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra daily bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )

    # Daily EMA34 trend filter (only needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above bearish fractal (resistance) with volume and uptrend
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i] and
                price_above_daily_ema and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bullish fractal (support) with volume and downtrend
            elif (not np.isnan(bullish_fractal_aligned[i]) and 
                  close[i] < bullish_fractal_aligned[i] and
                  price_below_daily_ema and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal (support) or trend turns down
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] < bullish_fractal_aligned[i]) or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal (resistance) or trend turns up
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i]) or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals