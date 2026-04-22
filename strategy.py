#1d_KAMA_Trend_RSI14_Filter_v1
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: KAMA trend + RSI14 filter + volume confirmation on 1d
    # KAMA adapts to volatility: reduces noise in sideways markets, follows in trends
    # RSI14 avoids overbought/oversold extremes, improves win rate
    # Volume confirms momentum behind the trend
    # Designed for 1d timeframe: low frequency, minimal fee decay
    # Works in bull/bear via adaptive KAMA and RSI filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix dimensions: volatility needs to align with change
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align length
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma20  # volume above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) AND RSI < 60 (not overbought) AND volume confirmation
            if close[i] > kama[i] and rsi[i] < 60 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) AND RSI > 40 (not oversold) AND volume confirmation
            elif close[i] < kama[i] and rsi[i] > 40 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or RSI extreme
            if position == 1:
                if close[i] < kama[i] or rsi[i] > 70:  # trend break or overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or rsi[i] < 30:  # trend break or oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI14_Filter_v1"
timeframe = "1d"
leverage = 1.0