#!/usr/bin/env python3
# 1d_TrendFollow_1wEMA34_DipBuy
# Hypothesis: On 1d timeframe, enter long when price dips to or below weekly EMA34 in a bullish regime (price > weekly EMA200),
# and exit when price returns to or above weekly EMA34. Reverse for short in bearish regime (price < weekly EMA200).
# Uses weekly EMA for trend and dynamic support/resistance to capture trends while avoiding whipsaws.
# Targets 10-25 trades/year for low fee drift, works in both bull and bear markets via regime filter.

name = "1d_TrendFollow_1wEMA34_DipBuy"
timeframe = "1d"
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
    
    # Load weekly data for EMA34 and EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA34 and EMA200
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure weekly EMA200 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema34_w = ema34_1w_aligned[i]
        ema200_w = ema200_1w_aligned[i]
        
        if position == 0:
            # LONG: Price at or below weekly EMA34 AND weekly trend bullish (price > weekly EMA200)
            if low[i] <= ema34_w and close[i] > ema200_w:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above weekly EMA34 AND weekly trend bearish (price < weekly EMA200)
            elif high[i] >= ema34_w and close[i] < ema200_w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to or above weekly EMA34 (trend resumption)
            if high[i] >= ema34_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or below weekly EMA34 (trend resumption)
            if low[i] <= ema34_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals