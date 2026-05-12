#!/usr/bin/env python3
# 6h_Liquidity_Imbalance_Reversal_1dTrend
# Hypothesis: Identify liquidity imbalances (wick-based imbalance) on 6h timeframe, 
# then trade reversals when price returns to fill the imbalance, filtered by 1d trend.
# Works in both bull and bear markets by fading short-term overextensions.
# Targets 15-25 trades/year by requiring imbalance formation, trend alignment, and mean reversion.

name = "6h_Liquidity_Imbalance_Reversal_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mft_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h liquidity imbalances (wick imbalance)
    # Bullish imbalance: low[i] > high[i-2] (gap up with buying pressure)
    # Bearish imbalance: high[i] < low[i-2] (gap down with selling pressure)
    bullish_imb = np.zeros(n, dtype=bool)
    bearish_imb = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        bullish_imb[i] = low[i] > high[i-2]  # Gap up
        bearish_imb[i] = high[i] < low[i-2]  # Gap down
    
    # Align imbalance signals to current bar (no look-ahead)
    bullish_imb_aligned = align_htf_to_ltf(prices, prices, bullish_imb.astype(float))
    bearish_imb_aligned = align_htf_to_ltf(prices, prices, bearish_imb.astype(float))
    
    # Mean reversion entry: price closes back into imbalance zone
    bullish_reversal = bullish_imb_aligned & (close <= high.shift(2).values)  # Close back below gap high
    bearish_reversal = bearish_imb_aligned & (close >= low.shift(2).values)   # Close back above gap low

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(bullish_reversal[i]) or np.isnan(bearish_reversal[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from daily EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: bullish imbalance filled with downtrend (fade the gap up)
            if bullish_reversal[i] and price_below_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish imbalance filled with uptrend (fade the gap down)
            elif bearish_reversal[i] and price_above_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches opposite imbalance or trend turns up
            if bearish_reversal[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches opposite imbalance or trend turns down
            if bullish_reversal[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals