#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter
# Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# RSI(14) < 30 for long, > 70 for short provides mean reversion edge
# 4h EMA50 trend filter: only long when price > EMA50, short when price < EMA50
# Session filter (08-20 UTC) reduces noise and whipsaws
# Discrete position sizing: 0.20 (20% of capital) manages risk in volatile markets
# Works in bull markets via pullbacks in uptrend and bear markets via bounces in downtrend

name = "1h_RSI14_4hEMA50_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(14) - primary signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate 4h EMA50 trend filter - HTF direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for RSI)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: RSI < 30 (oversold) AND price > 4h EMA50 (bullish trend)
            if (rsi[i] < 30 and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 70 (overbought) AND price < 4h EMA50 (bearish trend)
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) OR price < 4h EMA50 (trend change)
            if rsi[i] > 50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) OR price > 4h EMA50 (trend change)
            if rsi[i] < 50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals