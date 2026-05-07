#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI14_StochasticFilter
# Hypothesis: KAMA adapts to market regime (trending/ranging) to avoid false signals.
# In trending markets: KAMA follows price closely, RSI > 50 confirms bullish momentum.
# In ranging markets: KAMA flattens, Stochastic oscillator identifies mean-reversion at extremes.
# Weekly trend filter ensures alignment with higher-timeframe direction.
# Position size 0.25 to manage risk during drawdowns.
# Target: 10-25 trades per year (~40-100 over 4 years).

name = "1d_KAMA_Trend_RSI14_StochasticFilter"
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
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
    
    # Handle array shapes for ER calculation
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    
    for i in range(13, n):
        lowest_low[i] = np.min(low[i-13:i+1])
        highest_high[i] = np.max(high[i-13:i+1])
    
    stoch_k = np.where((highest_high - lowest_low) != 0, 
                       (close - lowest_low) / (highest_high - lowest_low) * 100, 50)
    stoch_d = np.full(n, np.nan)
    
    for i in range(15, n):
        stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or
            np.isnan(ema_10_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA slope for trend direction
        kama_slope = kama[i] - kama[i-1]
        
        # Trend filter from weekly EMA10
        uptrend = close[i] > ema_10_1w_aligned[i]
        downtrend = close[i] < ema_10_1w_aligned[i]
        
        if position == 0:
            # Long: price above rising KAMA + RSI > 50 + Stochastic oversold
            if (close[i] > kama[i] and kama_slope > 0 and 
                rsi[i] > 50 and stoch_k[i] < 30 and stoch_d[i] < 30 and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price below falling KAMA + RSI < 50 + Stochastic overbought
            elif (close[i] < kama[i] and kama_slope < 0 and 
                  rsi[i] < 50 and stoch_k[i] > 70 and stoch_d[i] > 70 and downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI < 40 or Stochastic overbought
            if (close[i] < kama[i] or rsi[i] < 40 or 
                stoch_k[i] > 70 or stoch_d[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI > 60 or Stochastic oversold
            if (close[i] > kama[i] or rsi[i] > 60 or 
                stoch_k[i] < 30 or stoch_d[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals