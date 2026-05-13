#!/usr/bin/env python3
# 6h_TurtleSoup_Reversal_1dTrend
# Hypothesis: Turtle Soup pattern (false breakout reversal) on 6h timeframe filtered by 1d trend direction.
# In uptrend: wait for false breakdown below prior 3-bar low, then go long on reversal above that low.
# In downtrend: wait for false breakout above prior 3-bar high, then go short on reversal below that high.
# This captures liquidity hunts where stops are hunted before real moves resume.
# Works in bull (buy false breakdowns) and bear (sell false breakouts).
# Low frequency due to pattern specificity and trend filter.

name = "6h_TurtleSoup_Reversal_1dTrend"
timeframe = "6h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if trend value is NaN
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Need at least 3 bars of history for pattern
            if i >= 3:
                # Calculate prior 3-bar low and high (excluding current bar)
                prior_low = min(low[i-3], low[i-2], low[i-1])
                prior_high = max(high[i-3], high[i-2], high[i-1])

                # Turtle Soup Long: false breakdown below 3-bar low, then reversal above it
                # Only in uptrend (price above daily EMA50)
                if (low[i] < prior_low and  # false breakdown
                    close[i] > prior_low and  # reversal back above
                    close[i] > ema50_1d_aligned[i]):  # uptrend filter
                    signals[i] = 0.25
                    position = 1

                # Turtle Soup Short: false breakout above 3-bar high, then reversal below it
                # Only in downtrend (price below daily EMA50)
                elif (high[i] > prior_high and  # false breakout
                      close[i] < prior_high and  # reversal back below
                      close[i] < ema50_1d_aligned[i]):  # downtrend filter
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0

        elif position == 1:
            # Exit long: close below 3-bar low (pattern failed) or trend reversal
            if i >= 3:
                prior_low = min(low[i-3], low[i-2], low[i-1])
                if close[i] < prior_low or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25

        elif position == -1:
            # Exit short: close above 3-bar high (pattern failed) or trend reversal
            if i >= 3:
                prior_high = max(high[i-3], high[i-2], high[i-1])
                if close[i] > prior_high or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25

    return signals