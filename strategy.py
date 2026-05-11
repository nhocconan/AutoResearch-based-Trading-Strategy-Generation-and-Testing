#!/usr/bin/env python3
"""
4h_Supertrend_RSI_MeanReversion_1dTrend
Hypothesis: Mean-reversion entries in trending markets using Supertrend for direction and RSI for timing.
In up-trend: go long when RSI < 30 (oversold). In down-trend: go short when RSI > 70 (overbought).
Uses 1d trend filter to ensure alignment with higher timeframe direction.
Designed for low trade frequency (<25/year) to minimize fee drag on 4h timeframe.
"""

name = "4h_Supertrend_RSI_MeanReversion_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Calculate Supertrend (10, 3.0) ===
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.zeros(n)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic Bands
    basic_ub = (high + low) / 2 + atr_multiplier * atr
    basic_lb = (high + low) / 2 - atr_multiplier * atr
    
    # Final Bands
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, n):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(n)
    supertrend[0] = final_ub[0]
    for i in range(1, n):
        if supertrend[i-1] == final_ub[i-1]:
            if close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend (price above supertrend), -1 for downtrend
    trend_dir = np.where(close > supertrend, 1, -1)
    
    # === Calculate RSI (14) ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1-day EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(atr_period, rsi_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema34_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: In uptrend (close > supertrend) and RSI oversold (<30) and 1d trend up
            if (trend_dir[i] == 1 and 
                rsi[i] < 30 and 
                close[i] > ema34_1d_4h[i]):
                signals[i] = position_size
                position = 1
            # Short: In downtrend (close < supertrend) and RSI overbought (>70) and 1d trend down
            elif (trend_dir[i] == -1 and 
                  rsi[i] > 70 and 
                  close[i] < ema34_1d_4h[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI returns to neutral territory (40-60) or trend changes
            if position == 1:
                if rsi[i] > 40 or trend_dir[i] == -1:  # Exit long if RSI recovers or trend turns down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi[i] < 60 or trend_dir[i] == 1:  # Exit short if RSI declines or trend turns up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals