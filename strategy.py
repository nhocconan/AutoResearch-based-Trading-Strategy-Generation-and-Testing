#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA Trend with 1d RSI and Weekly Volatility Regime Filter
# Uses Kaufman Adaptive Moving Average (KAMA) to detect trend direction
# Enters long when price > KAMA and RSI < 70 (avoiding overbought)
# Enters short when price < KAMA and RSI > 30 (avoiding oversold)
# Filters trades using weekly ATR-based volatility regime: only trade when weekly ATR ratio > 0.8
# Designed to capture trends while avoiding extreme readings and low volatility periods
# Target: 15-30 trades per symbol over 4 years (4-7.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (10-period ER, 2/30 for fast/slow SC)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Weekly ATR (14-period) and its 50-period average for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_w = pd.Series(tr_w).ewm(span=14, adjust=False).mean().values
    atr_ma_w = pd.Series(atr_w).ewm(span=50, adjust=False).mean().values
    atr_ratio_w = atr_w / atr_ma_w
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_ratio_w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_ratio_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price above KAMA, RSI not overbought, sufficient volatility
            if (price > kama_aligned[i] and 
                rsi_aligned[i] < 70 and 
                atr_ratio_w_aligned[i] > 0.8):
                position = 1
                signals[i] = position_size
            # Short setup: price below KAMA, RSI not oversold, sufficient volatility
            elif (price < kama_aligned[i] and 
                  rsi_aligned[i] > 30 and 
                  atr_ratio_w_aligned[i] > 0.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if price < kama_aligned[i] or rsi_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if price > kama_aligned[i] or rsi_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_WeeklyVolatilityRegime"
timeframe = "1d"
leverage = 1.0