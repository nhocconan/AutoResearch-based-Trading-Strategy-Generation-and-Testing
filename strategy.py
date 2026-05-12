#!/usr/bin/env python3
# 4H_RSI_MEAN_REVERSION_1D_TREND_FILTER
# Hypothesis: RSI mean-reversion works best when aligned with the daily trend.
# In 1d uptrend (price > EMA50), go long when RSI(14) < 30 (oversold).
# In 1d downtrend (price < EMA50), go short when RSI(14) > 70 (overbought).
# Trend filter prevents counter-trend trades, RSI captures reversals within trend.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_RSI_MEAN_REVERSION_1D_TREND_FILTER"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + RSI oversold
            if (close[i] > ema50_aligned[i] and 
                rsi[i] < 30):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + RSI overbought
            elif (close[i] < ema50_aligned[i] and 
                  rsi[i] > 70):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or RSI overbought
            if (close[i] <= ema50_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or RSI oversold
            if (close[i] >= ema50_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals