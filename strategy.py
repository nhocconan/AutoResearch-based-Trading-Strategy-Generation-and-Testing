#!/usr/bin/env python3
"""
1d_WeeklyTrend_Pullback_v1
Hypothesis: On daily timeframe, buy pullbacks in weekly uptrend and sell rallies in weekly downtrend.
Uses weekly EMA for trend direction and daily RSI for mean-reversion entries.
Designed for low trade frequency (10-25/year) to minimize fee drag in both bull and bear markets.
"""

name = "1d_WeeklyTrend_Pullback_v1"
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
    
    # Weekly trend: EMA(21) of weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily RSI(14) for mean-reversion entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend + RSI oversold (mean reversion)
            if weekly_uptrend and rsi_values[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI overbought (mean reversion)
            elif weekly_downtrend and rsi_values[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: weekly trend turns against position OR RSI overbought
            if not weekly_uptrend or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: weekly trend turns against position OR RSI oversold
            if not weekly_downtrend or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals