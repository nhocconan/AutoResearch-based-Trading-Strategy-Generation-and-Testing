#!/usr/bin/env python3
# 1h_4h1d_rsi_ema_pullback_v1
# Hypothesis: Pullback to EMA21 on 4h/1d trend with RSI(14) oversold/overbought on 1h for entry.
# Trend: EMA21 on 4h and 1d must agree (both bullish or both bearish).
# Entry: In uptrend, go long when 1h RSI < 30 and price > 4h EMA21; in downtrend, go short when 1h RSI > 70 and price < 4h EMA21.
# Exit: When RSI returns to neutral (40-60 range) or trend changes.
# Works in bull markets by buying dips in uptrend; in bear markets by selling rallies in downtrend.
# Target: 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_rsi_ema_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA21 for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend alignment: both 4h and 1d EMA21 must agree
        uptrend = (close[i] > ema_4h_aligned[i]) and (close[i] > ema_1d_aligned[i])
        downtrend = (close[i] < ema_4h_aligned[i]) and (close[i] < ema_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] >= 40 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] <= 60 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: uptrend + RSI oversold + in session
            if uptrend and rsi[i] < 30 and in_session:
                position = 1
                signals[i] = 0.20
            # Enter short: downtrend + RSI overbought + in session
            elif downtrend and rsi[i] > 70 and in_session:
                position = -1
                signals[i] = -0.20
    
    return signals