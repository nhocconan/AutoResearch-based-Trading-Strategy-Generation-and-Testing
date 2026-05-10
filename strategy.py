#!/usr/bin/env python3
# 1D_RSI_Trend_Filter
# Hypothesis: Uses daily RSI(14) filtered by weekly trend (price > weekly EMA50) for long entries
# and (price < weekly EMA50) for short entries. RSI < 30 triggers long, RSI > 70 triggers short.
# Exits when RSI returns to neutral zone (40-60). Uses weekly EMA50 to avoid counter-trend trades
# and improve performance in both bull and bear markets. Targets 7-25 trades per year on 1d timeframe.

name = "1D_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    def rma(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*period)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) in uptrend (price > weekly EMA50)
            if (rsi[i] < 30 and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 70 (overbought) in downtrend (price < weekly EMA50)
            elif (rsi[i] > 70 and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral zone (>= 40) or trend changes
            if (rsi[i] >= 40 or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral zone (<= 60) or trend changes
            if (rsi[i] <= 60 or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals