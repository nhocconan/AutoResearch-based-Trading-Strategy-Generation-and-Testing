#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback + 4h/1d Trend Filter
# Hypothesis: During pullbacks in strong trends (4h EMA20 > EMA50 AND 1d close > SMA50),
# buy RSI < 30 and sell RSI > 70. Works in bull/bear by only trading with trend.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_rsi_pullback_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI (14) on 1h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(20) and EMA(50) for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d SMA(50) for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA20 > EMA50 AND 1d close > SMA50
        uptrend = ema20_4h_aligned[i] > ema50_4h_aligned[i] and close[i] > sma50_1d_aligned[i]
        downtrend = ema20_4h_aligned[i] < ema50_4h_aligned[i] and close[i] < sma50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or trend changes
            if rsi[i] > 70 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 or trend changes
            if rsi[i] < 30 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if uptrend and rsi[i] < 30:
                # Long: pullback in uptrend
                position = 1
                signals[i] = 0.20
            elif downtrend and rsi[i] > 70:
                # Short: pullback in downtrend
                position = -1
                signals[i] = -0.20
    
    return signals