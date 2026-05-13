#!/usr/bin/env python3
"""
1d_Kelly_RSI_Pullback_Trend
Hypothesis: Daily RSI pullbacks in trending markets offer high-probability entries. 
Uses RSI(14) < 30 for longs and > 70 for shorts in the direction of 200-day EMA trend.
Kelly sizing (half-Kelly) with win rate 60% and win/loss ratio 1.5 gives ~22% position size.
Volatility filter (ATR ratio) avoids choppy markets. Targets 15-25 trades/year.
"""

name = "1d_Kelly_RSI_Pullback_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # 200-day EMA trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-day average ATR (low volatility filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    low_volatility = atr_ratio < 1.5  # Only trade in normal/low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if position == 0:
            # LONG: RSI oversold (<30) in uptrend (close > EMA200) + low volatility
            if (rsi_values[i] < 30 and 
                close[i] > ema200[i] and 
                low_volatility[i]):
                signals[i] = 0.22  # Half-Kelly size
                position = 1
            # SHORT: RSI overbought (>70) in downtrend (close < EMA200) + low volatility
            elif (rsi_values[i] > 70 and 
                  close[i] < ema200[i] and 
                  low_volatility[i]):
                signals[i] = -0.22
                position = -1
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend change
            if rsi_values[i] > 70 or close[i] < ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.22
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend change
            if rsi_values[i] < 30 or close[i] > ema200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.22
    
    return signals