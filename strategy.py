#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h/1d Trend Filter
# Hypothesis: In trending markets (4h/1d aligned), pullbacks to RSI(14) < 30 in uptrend or > 70 in downtrend offer high-probability entries.
# Uses 1h for precise entry timing, 4h/1d for trend direction. Works in bull by buying dips, in bear by selling rallies.
# Session filter (08-20 UTC) reduces noise. Target: 15-35 trades/year (60-140 total over 4 years).

name = "1h_rsi_pullback_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend is up if both 4h and 1d EMA(50) sloping up
        trend_up = (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1])
        # Trend is down if both sloping down
        trend_down = (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1])
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns down
            if rsi[i] > 70 or trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns up
            if rsi[i] < 30 or trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Enter long: RSI < 30 (oversold) and trend up
            if rsi[i] < 30 and trend_up:
                position = 1
                signals[i] = 0.20
            # Enter short: RSI > 70 (overbought) and trend down
            elif rsi[i] > 70 and trend_down:
                position = -1
                signals[i] = -0.20
    
    return signals