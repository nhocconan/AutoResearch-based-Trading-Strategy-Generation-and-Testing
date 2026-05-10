#!/usr/bin/env python3
# 6H_Aroon_Trend_With_WilliamsR_OverboughtOversold_Filter
# Hypothesis: Aroon indicator identifies strong trends (Aroon Up > 70 or Aroon Down > 70).
# Williams %R filters overextended entries: only go long when Williams %R < -20 (not oversold),
# and short when Williams %R > -80 (not overbought). This avoids chasing extremes.
# Weekly trend filter (price above/below weekly EMA20) ensures alignment with higher timeframe.
# Designed for low trade frequency (~15-30/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (catching strong uptrends) and bear markets (catching strong downtrends).

name = "6H_Aroon_Trend_With_WilliamsR_OverboughtOversold_Filter"
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
    
    # Aroon indicator (25-period)
    def aroon_up(high, lookback=25):
        h = pd.Series(high)
        return h.rolling(window=lookback, min_periods=lookback).apply(
            lambda x: (lookback - 1 - np.argmax(x)) / (lookback - 1) * 100, raw=True
        ).values
    
    def aroon_down(low, lookback=25):
        l = pd.Series(low)
        return l.rolling(window=lookback, min_periods=lookback).apply(
            lambda x: (lookback - 1 - np.argmin(x)) / (lookback - 1) * 100, raw=True
        ).values
    
    aroon_up_val = aroon_up(high, 25)
    aroon_down_val = aroon_down(low, 25)
    
    # Williams %R (14-period)
    def williams_r(high, low, close, lookback=14):
        h = pd.Series(high)
        l = pd.Series(low)
        c = pd.Series(close)
        highest_high = h.rolling(window=lookback, min_periods=lookback).max()
        lowest_low = l.rolling(window=lookback, min_periods=lookback).min()
        wr = -100 * (highest_high - c) / (highest_high - lowest_low)
        return wr.values
    
    wr = williams_r(high, low, close, 14)
    
    # Weekly trend filter: EMA 20
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly close for trend
    close_1w_series = pd.Series(close_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_series.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up_val[i]) or np.isnan(aroon_down_val[i]) or np.isnan(wr[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_weekly_uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        is_weekly_downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: Aroon Up > 70 (strong uptrend) + Williams %R > -80 (not overbought) + weekly uptrend
            if aroon_up_val[i] > 70 and wr[i] > -80 and is_weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Aroon Down > 70 (strong downtrend) + Williams %R < -20 (not oversold) + weekly downtrend
            elif aroon_down_val[i] > 70 and wr[i] < -20 and is_weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Aroon Down > 70 (strong downtrend emerges) or weekly trend turns down
            if aroon_down_val[i] > 70 or not is_weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Aroon Up > 70 (strong uptrend emerges) or weekly trend turns up
            if aroon_up_val[i] > 70 or not is_weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals