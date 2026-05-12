#!/usr/bin/env python3
# 1d_KAMA_Signal_1wTrend_PriceAction
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 1d to determine trend direction, with 1w EMA34 as higher timeframe trend filter. Enter long when KAMA turns upward and price is above 1w EMA34, enter short when KAMA turns downward and price is below 1w EMA34. Use price action confirmation (close > open for longs, close < open for shorts) to avoid whipsaws. Designed for low frequency (10-25 trades/year) to capture major trends while minimizing false signals in choppy markets.

name = "1d_KAMA_Signal_1wTrend_PriceAction"
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
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1w EMA34 for higher timeframe trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA on 1d (fast=2, slow=30) ===
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(kama[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA34
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        # KAMA direction: turning point
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Price action confirmation
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # LONG: KAMA turning up, price above 1w EMA34, bullish candle
            if kama_up and trend_up and bullish_candle:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down, price below 1w EMA34, bearish candle
            elif kama_down and trend_down and bearish_candle:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA turns down or trend breaks
            if not kama_up or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or trend breaks
            if not kama_down or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals