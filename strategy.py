#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI with Weekly Trend Filter
# Uses daily KAMA for trend direction, RSI(14) for mean-reversion entries, and weekly EMA for trend filter
# KAMA adapts to market noise, reducing whipsaws in ranging markets
# RSI provides entry signals in direction of trend with mean-reversion pullbacks
# Weekly EMA ensures alignment with higher timeframe trend to avoid counter-trend trades
# Works in bull/bear by only taking RSI reversals in direction of weekly trend
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency ratio = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w[i])):
            signals[i] = 0.0
            continue
        
        # Align weekly indicators to daily
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        
        price = close[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > KAMA and above weekly EMA20
            if (rsi[i] < 30 and price > kama[i] and 
                price > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) and price < KAMA and below weekly EMA20
            elif (rsi[i] > 70 and price < kama[i] and 
                  price < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or price < KAMA
            if rsi[i] > 50 or price < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or price > KAMA
            if rsi[i] < 50 or price > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0