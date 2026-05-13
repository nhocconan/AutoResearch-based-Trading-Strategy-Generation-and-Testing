#!/usr/bin/env python3
# 6h_MixedCandlestick_Pattern_Confirmation
# Hypothesis: Combine bullish/bearish engulfing patterns with 1d RSI extremes and 1w trend filter.
# Engulfing patterns signal potential reversals; RSI <30 or >70 confirms overextension; 1w EMA200 filters trend direction.
# Works in bull via buying oversold bounces in uptrend, bear via selling overbought pullbacks in downtrend.
# Target: 20-40 trades/year by requiring pattern + RSI extreme + trend alignment.

name = "6h_MixedCandlestick_Pattern_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Bullish engulfing: current bullish candle engulfs previous bearish candle
    bullish_engulf = (close > open_) & (open_ < close) & (close[-1:] > open_[:-1]) & (open_[-1:] < close[:-1])
    bullish_engulf = np.concatenate([[False], bullish_engulf[:-1]])  # shift for previous candle comparison

    # Bearish engulfing: current bearish candle engulfs previous bullish candle
    bearish_engulf = (close < open_) & (open_ > close) & (close[-1:] < open_[:-1]) & (open_[-1:] > close[:-1])
    bearish_engulf = np.concatenate([[False], bearish_engulf[:-1]])

    # 1d RSI(14) for overbought/oversold conditions
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_ltf_to_htf(prices, df_1d, rsi_1d)

    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_ltf_to_htf(prices, df_1w, ema200_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish engulfing + RSI < 30 (oversold) + price > 1w EMA200 (uptrend)
            if (bullish_engulf[i] and 
                rsi_1d_aligned[i] < 30 and
                close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish engulfing + RSI > 70 (overbought) + price < 1w EMA200 (downtrend)
            elif (bearish_engulf[i] and 
                  rsi_1d_aligned[i] > 70 and
                  close[i] < ema200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish engulfing or RSI > 50 (momentum fade)
            if bearish_engulf[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish engulfing or RSI < 50 (momentum fade)
            if bullish_engulf[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

def align_ltf_to_htf(ltf_prices, htf_df, htf_values):
    """Wrapper for align_htf_to_ltf with clearer naming"""
    from mtf_data import align_htf_to_ltf
    return align_htf_to_ltf(ltf_prices, htf_df, htf_values)