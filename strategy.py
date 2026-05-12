#!/usr/bin/env python3
# 6H_RSI_OVERBOUGHT_OVERSOLD_1D_TREND_FILTER
# Hypothesis: RSI(14) overbought/oversold levels on 6h chart, filtered by 1d EMA trend, capture mean reversion moves.
# Works in both bull and bear markets: in uptrends, buy oversold pullbacks; in downtrends, sell overbought bounces.
# Target: 20-40 trades/year on 6h timeframe.

name = "6H_RSI_OVERBOUGHT_OVERSOLD_1D_TREND_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 for trend filter
    ema20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    
    # Calculate RSI(14) on 6h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need RSI warmup
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) in uptrend (price above EMA20)
            if rsi[i] < 30 and close[i] > ema20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) in downtrend (price below EMA20)
            elif rsi[i] > 70 and close[i] < ema20_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or trend reversal
            if rsi[i] > 70 or close[i] <= ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or trend reversal
            if rsi[i] < 30 or close[i] >= ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals