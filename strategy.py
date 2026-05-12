#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_Volume_Squeeze
# Hypothesis: Keltner Channel breakout with Bollinger Band squeeze filter and volume confirmation on 4h timeframe.
# Uses Keltner Channel (ATR-based) for breakout detection, Bollinger Band width for volatility regime (squeeze = low volatility),
# and volume spike for confirmation. Works in both bull and bear markets by following breakout direction.
# Low volatility squeeze often precedes strong moves, reducing false breakouts and whipsaw.

name = "4h_Keltner_Channel_Breakout_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for Keltner Channel and Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values

    # Calculate ATR for Keltner Channel
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Keltner Channel: EMA(20) ± ATR * 2
    ema20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr

    # Bollinger Bands for squeeze detection
    sma20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20  # Normalized width

    # Bollinger Band width percentile for squeeze (low volatility = squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).rank(pct=True).values

    # Align Keltner Channel and Bollinger Band width percentile to lower timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_4h, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_4h, kc_lower)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile)

    # Calculate volume spike threshold (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner upper band during low volatility squeeze with volume spike
            if (close[i] > kc_upper_aligned[i] and 
                bb_width_percentile_aligned[i] < 0.3 and  # Squeeze: bottom 30% of BB width
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner lower band during low volatility squeeze with volume spike
            elif (close[i] < kc_lower_aligned[i] and 
                  bb_width_percentile_aligned[i] < 0.3 and  # Squeeze: bottom 30% of BB width
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Keltner lower band (mean reversion) or volatility expansion
            if close[i] < kc_lower_aligned[i] or bb_width_percentile_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Keltner upper band (mean reversion) or volatility expansion
            if close[i] > kc_upper_aligned[i] or bb_width_percentile_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals