#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Trend_Filter
# Hypothesis: Use KAMA to detect trend direction on daily timeframe, combined with RSI for momentum and trend filter (EMA200) for higher probability trades.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI avoids overbought/oversold extremes.
# Designed for 1d to capture multi-day trends with low trade frequency, suitable for both bull and bear markets.

name = "1d_KAMA_Direction_RSI_Trend_Filter"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        vol = np.sum(np.abs(np.diff(close, prepend=close[0]))[:len(close)])
        er = np.zeros_like(close)
        for i in range(len(close)):
            if vol[i] != 0:
                er[i] = change[i] / vol[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[0] = np.mean(gain[:length]) if len(gain) >= length else 0
        avg_loss[0] = np.mean(loss[:length]) if len(loss) >= length else 0
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    # EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate indicators
    kama_val = kama(close, er_len=10, fast_len=2, slow_len=30)
    rsi_val = rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough history for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(ema_200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, and above EMA200
            if close[i] > kama_val[i] and rsi_val[i] < 70 and close[i] > ema_200[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, and below EMA200
            elif close[i] < kama_val[i] and rsi_val[i] > 30 and close[i] < ema_200[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought
            if close[i] < kama_val[i] or rsi_val[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold
            if close[i] > kama_val[i] or rsi_val[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals