#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter
# Long when RSI < 30 AND 4h close > 4h EMA50 AND hour in [08,20] UTC
# Short when RSI > 70 AND 4h close < 4h EMA50 AND hour in [08,20] UTC
# Exit when RSI crosses 50 (mean reversion completion)
# Uses 1h primary timeframe with 4h HTF for trend filter to reduce whipsaw
# Session filter (08-20 UTC) reduces noise and fee drag
# Discrete sizing (0.20) to limit fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in bull markets via 4h trend filter and in bear markets via RSI mean reversion

name = "1h_RSI14_MeanReversion_4hEMA50_Trend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h data
    if len(close) >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 08-20
        
        if position == 0:
            # Long conditions: RSI oversold AND 4h close > 4h EMA50 AND in session
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI overbought AND 4h close < 4h EMA50 AND in session
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion completion)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion completion)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals