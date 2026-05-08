#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h direction filter (EMA21) and 1d momentum filter (RSI > 50 for long, < 50 for short)
# Uses 4h EMA for trend direction, 1d RSI for momentum confirmation, and 1h for precise entry timing
# Targets 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) to reduce noise
# Position size fixed at 0.20 to manage risk and avoid overtrading

name = "1h_EMA21_RSI50_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for momentum filter (RSI14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d_values = rsi14_1d.values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d_values)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # warmup for EMA21
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema21_4h_val = ema21_4h_aligned[i]
        rsi14_1d_val = rsi14_1d_aligned[i]
        
        if position == 0:
            # Enter long: price above 4h EMA21, 1d RSI > 50 (bullish momentum)
            if close_val > ema21_4h_val and rsi14_1d_val > 50:
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA21, 1d RSI < 50 (bearish momentum)
            elif close_val < ema21_4h_val and rsi14_1d_val < 50:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price below 4h EMA21 or 1d RSI < 50
            if close_val < ema21_4h_val or rsi14_1d_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price above 4h EMA21 or 1d RSI > 50
            if close_val > ema21_4h_val or rsi14_1d_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals