# 12h_KAMA_Trend_RSI_MeanReversion_v1
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter.
# In trending markets, price follows KAMA closely; in ranging markets, price mean-reverts around KAMA.
# Combined with RSI for overbought/oversold conditions and a 1d trend filter (EMA50) to avoid counter-trend trades.
# Timeframe: 12h reduces trade frequency to avoid fee drag; uses 1d HTF for trend alignment.
# Works in both bull (trend following) and bear (mean reversion in range) markets via adaptive KAMA and RSI extremes.
# Expected trades: 20-40/year, well within limits.

#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_MeanReversion_v1"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # === 12h KAMA (trend/adaptive) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute correctly below
    # Recompute volatility properly: sum of absolute changes over 10 periods
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === 12h RSI (mean reversion) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # after KAMA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price near KAMA (trend) + RSI oversold + above daily EMA50 (uptrend filter)
            if (abs(close[i] - kama[i]) / kama[i] < 0.02 and  # within 2% of KAMA
                rsi[i] < 30 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price near KAMA + RSI overbought + below daily EMA50 (downtrend filter)
            elif (abs(close[i] - kama[i]) / kama[i] < 0.02 and
                  rsi[i] > 70 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price far from KAMA (trend broken) or RSI overbought
            if (abs(close[i] - kama[i]) / kama[i] > 0.05 or  # >5% deviation
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price far from KAMA or RSI oversold
            if (abs(close[i] - kama[i]) / kama[i] > 0.05 or
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals