#!/usr/bin/env python3
# 1d_RSI2_Consensus_With_1wTrend_Filter
# Hypothesis: On daily timeframe, buy when RSI(2) < 10 (oversold) and sell when RSI(2) > 90 (overbought),
# but only in the direction of the weekly trend (EMA34). This captures mean-reversion
# within the prevailing trend, reducing false signals in strong trends.
# Extremely tight RSI thresholds yield low trade frequency, minimizing fee drag.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "1d_RSI2_Consensus_With_1wTrend_Filter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily RSI(2)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34) and RSI(2) period (2)
    start_idx = max(34, 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # RSI(2) extreme levels
        rsi_oversold = rsi_values[i] < 10
        rsi_overbought = rsi_values[i] > 90
        
        if position == 0:
            # Long entry: RSI(2) < 10 (oversold) + weekly uptrend
            if rsi_oversold and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI(2) > 90 (overbought) + weekly downtrend
            elif rsi_overbought and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion complete) or trend turns down
            if rsi_values[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion complete) or trend turns up
            if rsi_values[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals