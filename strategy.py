#!/usr/bin/env python3
# 1d_1w_momentum_pullback_v1
# Hypothesis: On daily timeframe, buy pullbacks in uptrend and sell rallies in downtrend using 200 EMA as trend filter and RSI for entry timing. 
# Weekly trend confirms direction to avoid counter-trend trades. Works in both bull/bear as trend adapts.
# Entry: Long when price > daily EMA200, weekly EMA200 rising, and RSI(14) pulls back from oversold (<30) to >30.
# Entry: Short when price < daily EMA200, weekly EMA200 falling, and RSI(14) rallies from overbought (>70) to <70.
# Exit: Opposite signal or RSI reaches extreme (70/30) to avoid mean-reversion traps.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_pullback_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily EMA200 for trend filter
    close_s = pd.Series(close)
    ema200_d = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_rising = ema200_1w > np.roll(ema200_1w, 1)
    ema200_1w_falling = ema200_1w < np.roll(ema200_1w, 1)
    ema200_1w_rising[0] = False
    ema200_1w_falling[0] = False
    
    # Align weekly EMA200 and its direction to daily
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w_rising.astype(float))
    ema200_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w_falling.astype(float))
    
    # RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema200_d[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI reaches overbought or trend turns
            if rsi[i] >= 70 or close[i] < ema200_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI reaches oversold or trend turns
            if rsi[i] <= 30 or close[i] > ema200_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above daily EMA200, weekly EMA200 rising, RSI pulling back from oversold
            if (close[i] > ema200_d[i] and 
                ema200_1w_rising_aligned[i] > 0.5 and 
                rsi[i-1] < 30 and rsi[i] >= 30):
                position = 1
                signals[i] = 0.25
            # Enter short: price below daily EMA200, weekly EMA200 falling, RSI pulling back from overbought
            elif (close[i] < ema200_d[i] and 
                  ema200_1w_falling_aligned[i] > 0.5 and 
                  rsi[i-1] > 70 and rsi[i] <= 70):
                position = -1
                signals[i] = -0.25
    
    return signals