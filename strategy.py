#!/usr/bin/env python3
# 6H_ELDER_RAY_BULL_POWER_1D_TREND_FILTER
# Hypothesis: Elder Ray Bull Power (close - EMA13) captures bullish energy, Bear Power (EMA13 - close) captures bearish energy.
# On 6h timeframe, we go long when Bull Power > 0 and 1d EMA34 trend is up, short when Bear Power > 0 and 1d EMA34 trend is down.
# Uses 1d trend filter to avoid counter-trend trades. Works in bull markets (riding strength) and bear markets (riding weakness).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6H_ELDER_RAY_BULL_POWER_1D_TREND_FILTER"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # EMA34 for 1d trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Elder Ray components on 6h: Bull Power = close - EMA13, Bear Power = EMA13 - close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    bull_power = close - ema13  # positive when close > EMA13 (bullish energy)
    bear_power = ema13 - close  # positive when close < EMA13 (bearish energy)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need EMA13 warmup
    
    for i in range(start_idx, n):
        # Skip if trend filter not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power positive (bullish energy) and 1d uptrend
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power positive (bearish energy) and 1d downtrend
            elif bear_power[i] > 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or trend reversal
            if bull_power[i] <= 0 or close[i] <= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative or trend reversal
            if bear_power[i] <= 0 or close[i] >= ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals