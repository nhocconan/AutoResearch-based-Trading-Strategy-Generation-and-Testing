#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter
# Hypothesis: Use daily KAMA (Kaufman Adaptive Moving Average) for trend direction on 1d timeframe,
# combined with RSI for momentum confirmation on 1d, and weekly EMA for higher timeframe trend filter.
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
# RSI filters out overextended moves. Weekly EMA ensures alignment with longer-term trend.
# Designed for low trade frequency (<25/year) to minimize fee drag in BTC/ETH.
# Works in both bull and bear markets by following adaptive trend with momentum confirmation.

name = "1d_KAMA_Trend_RSI_Filter"
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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs correction
    
    # Proper ER calculation
    er = np.full_like(close, np.nan)
    for i in range(er_period, len(close)):
        if i >= er_period:
            ch = np.abs(close[i] - close[i-er_period])
            vol = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
            if vol != 0:
                er[i] = ch / vol
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = np.full_like(close, np.nan)
    for i in range(len(close)):
        if not np.isnan(er[i]):
            fast_sc = 2 / (fast_ema + 1)
            slow_sc = 2 / (slow_ema + 1)
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        else:
            sc[i] = np.nan
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    if len(close) > er_period:
        kama[er_period] = np.mean(close[0:er_period+1])
        for i in range(er_period+1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, None, kama)  # Same timeframe, no alignment needed
    
    # Calculate daily RSI (14-period)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= rsi_period + 1:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rsi = np.full_like(close, np.nan)
    for i in range(rsi_period, len(close)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period+1, rsi_period, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI not overbought (<60) AND weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 60 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI not oversold (>40) AND weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 40 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI overbought (>70) OR weekly trend reversal
            if (close[i] < kama[i] or 
                rsi[i] > 70 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI oversold (<30) OR weekly trend reversal
            if (close[i] > kama[i] or 
                rsi[i] < 30 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals