#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with RSI(14) and 1w EMA(20) trend filter
# Long when KAMA is rising, RSI(14) > 50, and 1w EMA(20) > prior
# Short when KAMA is falling, RSI(14) < 50, and 1w EMA(20) < prior
# Uses KAMA for adaptive trend, RSI for momentum filter, 1w EMA for trend alignment
# Target: 30-100 total trades over 4 years with controlled risk in both bull and bear markets
# Uses 1d timeframe with 1w trend filter to reduce trade frequency and improve signal quality

name = "1d_kama_rsi_1w_ema_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # KAMA (adaptive moving average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if i > 0 and close[i] < entry_price - 2.0 * (abs(close[i] - close[i-1])):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns down or RSI < 50 or 1w EMA turns down
            elif kama[i] < kama[i-1] or rsi[i] < 50 or ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if i > 0 and close[i] > entry_price + 2.0 * (abs(close[i] - close[i-1])):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns up or RSI > 50 or 1w EMA turns up
            elif kama[i] > kama[i-1] or rsi[i] > 50 or ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment
            if kama[i] > kama[i-1] and rsi[i] > 50 and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif kama[i] < kama[i-1] and rsi[i] < 50 and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals