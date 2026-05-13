#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend filter. Combined with RSI for momentum confirmation and a volatility filter to avoid choppy markets, this strategy aims to capture strong trends while minimizing false signals. Designed for low trade frequency on daily timeframe to reduce fee drag and improve robustness across bull/bear cycles.
"""

name = "1d_KAMA_Direction_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on weekly close for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) > 1 else np.zeros(n)
    # For each point, calculate ER = |change| / volatility over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        vol_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))  # sum of abs changes over 10 periods
        er[i] = price_change / (vol_sum + 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # between 2 and 30 period EMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volatility filter: avoid trading when volatility is too high (choppy market)
    # Use ATR(20) normalized by price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_normalized = atr / close
    vol_filter = atr_normalized < 0.05  # Avoid when daily volatility > 5%
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA and RSI warmup
        if position == 0:
            # LONG: Price above KAMA (uptrend), RSI > 55 (bullish momentum), volatility filter OK, price above weekly EMA50
            if (close[i] > kama[i] and 
                rsi[i] > 55 and 
                vol_filter[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), RSI < 45 (bearish momentum), volatility filter OK, price below weekly EMA50
            elif (close[i] < kama[i] and 
                  rsi[i] < 45 and 
                  vol_filter[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40 (loss of momentum)
            if (close[i] < kama[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60 (loss of bearish momentum)
            if (close[i] > kama[i] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals