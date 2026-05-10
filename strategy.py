#!/usr/bin/env python3
"""
1h_RSI_40_60_MeanReversion_1dTrend_Filter
Hypothesis: In 1h timeframe, mean-reversion at RSI 40-60 levels works when aligned with 1d trend.
In bull markets: buy RSI<40 in uptrend, sell RSI>60 in uptrend (momentum continuation).
In bear markets: sell RSI>60 in downtrend, buy RSI<40 in downtrend (trend continuation).
Uses 1d EMA50 as trend filter to avoid counter-trend trades. Target: 20-40 trades/year.
"""

name = "1h_RSI_40_60_MeanReversion_1dTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) on 1h data
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # EMA + RSI warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_vals[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_vals[i]
        price = close[i]
        trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI < 40 (oversold) AND price above 1d EMA (uptrend)
            # OR RSI > 60 (overbought) AND price below 1d EMA (downtrend continuation short)
            if rsi_val < 40 and price > trend:
                signals[i] = 0.20
                position = 1
            elif rsi_val > 60 and price < trend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or trend breaks down
            if rsi_val > 60 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or trend breaks up
            if rsi_val < 40 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals