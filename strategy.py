#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_RSI_Momentum_v2
Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for momentum, and weekly trend filter for regime alignment.
KAMA adapts to market noise, reducing false signals in choppy markets. RSI confirms momentum strength.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
This combination should work in both bull (KAMA up, RSI>50) and bear (KAMA down, RSI<50) markets.
Target: 15-25 trades/year by requiring alignment of daily KAMA, RSI momentum, and weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_RSI_Momentum_v2"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first element
    volatility = np.concatenate([[np.sum(np.abs(np.diff(close[:2])))] if len(close) > 1 else [0], volatility[1:]])
    
    # Efficiency ratio
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # KAMA for trend (daily)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # RSI for momentum (daily)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for KAMA and RSI
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Daily conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry conditions: KAMA + RSI + weekly trend alignment
        long_entry = price_above_kama and rsi_bullish and weekly_uptrend
        short_entry = price_below_kama and rsi_bearish and weekly_downtrend
        
        # Exit conditions: opposite signal
        long_exit = price_below_kama or rsi_bearish
        short_exit = price_above_kama or rsi_bullish
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals