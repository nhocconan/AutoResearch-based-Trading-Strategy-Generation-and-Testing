#!/usr/bin/env python3
name = "12h_KAMA_Direction_1dRSI_Trend"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend and RSI filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA on 12h prices (trend detection)
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Daily RSI(14) for overbought/oversold
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align with df_1d index
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # wait for KAMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions: avoid extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Trend filter: price vs daily EMA50
        price_above_ema50 = close[i] > ema_50_1d_aligned[i]
        price_below_ema50 = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, uptrend
            if price_above_kama and rsi_not_overbought and price_above_ema50:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, downtrend
            elif price_below_kama and rsi_not_oversold and price_below_ema50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI overbought
            if price_below_kama or rsi_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI oversold
            if price_above_kama or rsi_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h KAMA trend with daily RSI and trend filter
# - KAMA adapts to market noise: tracks trend efficiently, reduces whipsaws
# - Long when price above KAMA, RSI < 70 (not overbought), and price > daily EMA50 (uptrend)
# - Short when price below KAMA, RSI > 30 (not oversold), and price < daily EMA50 (downtrend)
# - Daily timeframe filter ensures alignment with higher timeframe trend
# - RSI prevents entries at extremes, improving win rate
# - Works in both bull (KAMA up in uptrend) and bear (KAMA down in downtrend)
# - Position size 0.25 balances return and risk, targeting ~20-50 trades/year
# - Avoids overtrading by requiring multiple confluence factors
# - Uses daily RSI and EMA50 for robustness across market regimes
# - KAMA's adaptive nature reduces lag vs traditional MAs in trending markets
# - Designed to minimize false signals during sideways markets via RSI filters