#!/usr/bin/env python3
# 1d_Camarilla_Pivot_Reversal_Volume
# Hypothesis: Price reversals at Camarilla pivot levels (S1/S2 for long, R1/R2 for short) on daily timeframe, confirmed by volume spike (>2x 20-period average) and weekly trend filter (price above/below weekly EMA20). Works in bull/bear markets as reversals occur at key levels regardless of trend. Targets 15-25 trades/year to minimize fee drag.

name = "1d_Camarilla_Pivot_Reversal_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Camarilla pivot levels for each day using previous day's OHLC
    # Camarilla formulas: 
    # H4 = Close + 1.1 * (High - Low) * 1.1/2
    # H3 = Close + 1.1 * (High - Low) * 1.1/4
    # H2 = Close + 1.1 * (High - Low) * 1.1/6
    # H1 = Close + 1.1 * (High - Low) * 1.1/12
    # L1 = Close - 1.1 * (High - Low) * 1.1/12
    # L2 = Close - 1.1 * (High - Low) * 1.1/6
    # L3 = Close - 1.1 * (High - Low) * 1.1/4
    # L4 = Close - 1.1 * (High - Low) * 1.1/2
    # We use H3/H2 for resistance (short) and L3/L2 for support (long)
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day has no previous day
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_ = prev_high - prev_low
    # Avoid division by zero in calculations
    range_safe = np.where(range_ == 0, 1e-10, range_)
    
    # Calculate Camarilla levels
    H3 = prev_close + 1.1 * range_safe * 1.1 / 4
    H2 = prev_close + 1.1 * range_safe * 1.1 / 6
    L3 = prev_close - 1.1 * range_safe * 1.1 / 4
    L2 = prev_close - 1.1 * range_safe * 1.1 / 6

    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(H3[i]) or np.isnan(H2[i]) or np.isnan(L3[i]) or np.isnan(L2[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches L3 or L2 (support) with volume spike
            # AND price is above weekly EMA20 (bullish weekly bias)
            if ((low[i] <= L3[i] or low[i] <= L2[i]) and
                close[i] > ema20_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches H3 or H2 (resistance) with volume spike
            # AND price is below weekly EMA20 (bearish weekly bias)
            elif ((high[i] >= H3[i] or high[i] >= H2[i]) and
                  close[i] < ema20_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches H3 (resistance) or weekly trend turns bearish
            if high[i] >= H3[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches L3 (support) or weekly trend turns bullish
            if low[i] <= L3[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals