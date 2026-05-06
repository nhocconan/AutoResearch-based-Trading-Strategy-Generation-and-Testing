# 1d_1wKAMA_Trend_Filtered_By_RSI
# Hypothesis: Use weekly trend via KAMA(30) on 1w timeframe to filter daily RSI(14) mean-reversion signals.
# Long when weekly trend is up (close > KAMA) and RSI < 30; short when weekly trend is down and RSI > 70.
# Weekly trend filter reduces whipsaws in choppy markets, allowing the strategy to work in both bull and bear trends.
# Target: 50-100 total trades over 4 years (12-25/year) with 0.25 position sizing to manage drawdown.

name = "1d_1wKAMA_Trend_Filtered_By_RSI"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    er = change / (volatility + 1e-10)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly KAMA(30)
    kama_1w = kama(close_1w, er_period=10, fast=2, slow=30)
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14)
    rsi_14 = rsi(close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly trend up (close > KAMA) and RSI oversold
            if close[i] > kama_1w_aligned[i] and rsi_14[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down (close < KAMA) and RSI overbought
            elif close[i] < kama_1w_aligned[i] and rsi_14[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend turns down or RSI overbought
            if close[i] < kama_1w_aligned[i] or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns up or RSI oversold
            if close[i] > kama_1w_aligned[i] or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals