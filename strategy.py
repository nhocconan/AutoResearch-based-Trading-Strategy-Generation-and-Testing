#!/usr/bin/env python3
"""
4h_RSI2_TrendFilter_12hVolatility
Hypothesis: RSI(2) identifies extreme short-term reversals, filtered by 12h EMA50 trend direction and 12h ATR-based volatility filter.
In strong trends, RSI(2) extremes offer high-probability pullback entries. Volatility filter avoids ranging markets.
Works in bull (buy RSI<10 in uptrend) and bear (sell RSI>90 in downtrend). Target: 80-150 total trades over 4 years (20-38/year).
"""

name = "4h_RSI2_TrendFilter_12hVolatility"
timeframe = "4h"
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
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), np.abs(low_12h - np.roll(close_12h, 1))))
    tr12h[0] = high_12h[0] - low_12h[0]
    atr14_12h = np.full(len(tr12h), np.nan)
    if len(tr12h) >= 14:
        atr14_12h[13] = np.mean(tr12h[:14])
        for i in range(14, len(tr12h)):
            atr14_12h[i] = (atr14_12h[i-1] * 13 + tr12h[i]) / 14
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # RSI(2) on 4h close
    def rsi(close, length):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        if len(gain) >= length:
            avg_gain[length-1] = np.mean(gain[:length])
            avg_loss[length-1] = np.mean(loss[:length])
            for i in range(length, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        return 100 - (100 / (1 + rs))
    
    rsi2 = rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 2)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_12h_aligned[i]) or np.isnan(rsi2[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility ranging markets
        vol_filter = atr14_12h_aligned[i] > np.nanpercentile(atr14_12h_aligned[:i+1], 30)
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) in uptrend with sufficient volatility
            if rsi2[i] < 10 and close[i] > ema50_12h_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) in downtrend with sufficient volatility
            elif rsi2[i] > 90 and close[i] < ema50_12h_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI(2) > 50 (mean reversion) or trend reversal
            if rsi2[i] > 50 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI(2) < 50 (mean reversion) or trend reversal
            if rsi2[i] < 50 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals