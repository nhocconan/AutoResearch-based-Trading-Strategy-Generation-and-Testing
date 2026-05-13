#!/usr/bin/env python3
# 4h_Engulfing_Signal_With_Volume_Confirmation
# Hypothesis: Bullish/bearish engulfing candles on 4h timeframe indicate strong momentum reversals.
# Combined with volume confirmation (>1.5x 20-bar average) and trend filter (price vs EMA50),
# this strategy captures high-probability reversals in both bull and bear markets.
# Engulfing patterns are reliable reversal signals that work across market regimes.
# Uses 4h timeframe with 12h trend filter to reduce noise and improve signal quality.

name = "4h_Engulfing_Signal_With_Volume_Confirmation"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: volume > 1.5 * 20-period average (~2.5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20

    # Detect bullish and bearish engulfing patterns
    bullish_engulf = (close > open_) & (open_ < close) & (close > open_.shift(1)) & (open_ < close.shift(1))
    bearish_engulf = (close < open_) & (open_ > close) & (close < open_.shift(1)) & (open_ > close.shift(1))
    # Fix: define open_ variable
    open_ = prices['open'].values
    bullish_engulf = (close > open_) & (open_ < close) & (close > np.roll(open_, 1)) & (open_ < np.roll(close, 1))
    bearish_engulf = (close < open_) & (open_ > close) & (close < np.roll(open_, 1)) & (open_ > np.roll(close, 1))
    # Handle first bar
    bullish_engulf[0] = False
    bearish_engulf[0] = False

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish engulfing + uptrend (price > EMA50) + volume confirmation
            if bullish_engulf[i] and close[i] > ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish engulfing + downtrend (price < EMA50) + volume confirmation
            elif bearish_engulf[i] and close[i] < ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish engulfing or price breaks below EMA50
            if bearish_engulf[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish engulfing or price breaks above EMA50
            if bullish_engulf[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals