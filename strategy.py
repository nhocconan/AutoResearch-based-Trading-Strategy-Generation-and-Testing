#!/usr/bin/env python3
# 1d_KAMA_Trend_Filtered_RSI_Mean_Reversion_1wTrend
# Hypothesis: Uses daily KAMA for primary trend, RSI for mean-reversion entries, and weekly EMA for trend filter.
# Works in bull/bear: KAMA adapts to volatility, RSI captures pullbacks in trend, weekly trend filters counter-trend.
# Target: 15-25 trades/year to minimize fee drag while capturing significant moves.

name = "1d_KAMA_Trend_Filtered_RSI_Mean_Reversion_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily KAMA (adaptive trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Correct calculation for ER
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i == 0:
            er[i] = 0
        else:
            dir_move = np.abs(close[i] - close[i-9]) if i >= 9 else np.abs(close[i] - close[0])
            vol_sum = np.sum(np.abs(np.diff(close[max(0, i-9):i+1]))) if i >= 9 else np.sum(np.abs(np.diff(close[:i+1])))
            er[i] = dir_move / (vol_sum + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA to daily (already daily, but for consistency)
    kama_aligned = kama  # Already calculated on daily data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below KAMA (pullback in uptrend), RSI oversold, above weekly EMA50
            if close[i] < kama_aligned[i] and rsi[i] < 30 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above KAMA (pullback in downtrend), RSI overbought, below weekly EMA50
            elif close[i] > kama_aligned[i] and rsi[i] > 70 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses above KAMA or RSI overbought
            if close[i] > kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses below KAMA or RSI oversold
            if close[i] < kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals