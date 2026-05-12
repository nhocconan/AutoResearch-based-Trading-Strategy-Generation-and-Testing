#/usr/bin/env python3
# 1D_1W_KAMA_TREND_WITH_RSI_FILTER
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise—fast in trends, slow in ranges.
# On daily timeframe, use KAMA(10,2,30) for trend direction. Filter with weekly trend (EMA34 on 1w) to avoid counter-trend trades.
# Add RSI(14) on daily to avoid overextended entries. Enter on pullbacks to KAMA in trending markets.
# Works in bull markets (trend following) and bear markets (avoiding false signals via weekly filter).
# Target: 15-25 trades/year on 1d timeframe.

name = "1D_1W_KAMA_TREND_WITH_RSI_FILTER"
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
    
    # Daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA parameters: ER fast=2, slow=30
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # This needs correction
    
    # Correct efficiency ratio calculation
    dir = np.abs(np.diff(close_1d, n=30, prepend=close_1d[:30]))  # direction over 30 periods
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # incorrect, fixing below
    
    # Proper ER calculation
    change_t = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_t = np.zeros_like(close_1d)
    for i in range(30, len(close_1d)):
        volatility_t[i] = np.sum(np.abs(np.diff(close_1d[i-29:i+1])))
    
    # Simpler and correct ER calculation
    price_change = np.abs(close_1d[30:] - close_1d[:-30])
    volatility_sum = np.array([np.sum(np.abs(np.diff(close_1d[i-29:i+1]))) for i in range(29, len(close_1d))])
    er = np.zeros_like(close_1d)
    er[30:] = np.where(volatility_sum > 0, price_change / volatility_sum, 0)
    
    # Smooth ER with smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend filter: EMA34 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > KAMA (uptrend), RSI not overbought, weekly uptrend
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 70 and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (downtrend), RSI not oversold, weekly downtrend
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 30 and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or overextension
            if (close[i] <= kama_aligned[i] or 
                rsi_aligned[i] >= 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or overextension
            if (close[i] >= kama_aligned[i] or 
                rsi_aligned[i] <= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals