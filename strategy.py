#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.sum(volatility[1:], axis=0) if np.sum(volatility[1:]) > 0 else 1)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI(14) for overbought/oversold
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Choppiness Index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                      np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                 np.abs(low_1d[1:] - close_1d[:-1])))
    atr1 = np.concatenate([[np.nan], atr1])
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_1d = chop
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        if position == 0:
            # Enter long: price above KAMA, RSI not overbought, chop indicates trend (not too choppy)
            if (close[i] > kama_val and rsi_val < 70 and chop_val < 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI not oversold, chop indicates trend
            elif (close[i] < kama_val and rsi_val > 30 and chop_val < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI overbought OR chop too high (range)
            if (close[i] < kama_val or rsi_val > 70 or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI oversold OR chop too high
            if (close[i] > kama_val or rsi_val < 30 or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses 12-hour timeframe with KAMA trend filter, RSI momentum, and Choppiness Index regime filter.
# - Enters long when price above 1d KAMA, RSI < 70, and chop < 61.8 (trending market)
# - Enters short when price below 1d KAMA, RSI > 30, and chop < 61.8
# - Exits when price crosses KAMA, RSI becomes extreme, or market becomes too choppy (chop > 61.8)
# - KAML provides adaptive trend following that reduces whipsaw
# - RSI filter prevents buying into overbought conditions or selling into oversold
# - Chop filter ensures we only trade in trending markets, avoiding range-bound losses
# - Position size 0.25 balances risk and return while minimizing fee churn
# - Designed to work in both bull and bear markets by following 1d trend direction
# - Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag
# - Works on BTC and ETH as primary targets (not SOL-only)