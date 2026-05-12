#!/usr/bin/env python3
"""
1d_KAMA_Direction_With_RSI_and_Chop_Regime
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
Combined with RSI for momentum confirmation and Choppiness Index to filter regimes, this should work in bull/bear markets.
Timeframe: 1d to limit trades and avoid fee drag. Uses 1w HTF for trend context.
Target: 30-100 trades over 4 years (7-25/year) to stay within fee drag limits.
"""

name = "1d_KAMA_Direction_With_RSI_and_Chop_Regime"
timeframe = "1d"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(price)
        kama_vals[period] = price[period]
        for i in range(period+1, len(price)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i] - kama_vals[i-1])
        return kama_vals
    
    # RSI
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Choppiness Index
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        # Wilder's smoothing
        for i in range(period+1, len(atr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop_vals = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop_vals[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop_vals[i] = 50
        return chop_vals
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    chop_vals = chop(high, low, close, period=14)
    
    # Weekly trend filter: 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend), RSI > 50 (momentum), CHOP < 61.8 (trending), above weekly EMA50
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] > 50 and 
                chop_vals[i] < 61.8 and
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), RSI < 50 (momentum), CHOP < 61.8 (trending), below weekly EMA50
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] < 50 and 
                  chop_vals[i] < 61.8 and
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40 (loss of momentum)
            if (close[i] < kama_vals[i]) or (rsi_vals[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60 (loss of momentum)
            if (close[i] > kama_vals[i]) or (rsi_vals[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals