#!/usr/bin/env python3
"""
1h Momentum Reversal with 4h/1d Trend Filter and Volume Confirmation
Hypothesis: In trending markets, pullbacks to the 4h EMA(20) with momentum divergence (RSI <30/70) offer high-probability entries.
Volume confirms institutional participation. Session filter (08-20 UTC) reduces noise.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_reversal_4h1d_trend_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 4h and 1d data for trend alignment (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA(20) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require high volume
    
    # 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA50_1d
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend alignment: 4h EMA20 vs 1d EMA50
        uptrend = ema_4h_aligned[i] > ema_50_1d_aligned[i]
        downtrend = ema_4h_aligned[i] < ema_50_1d_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: momentum fading OR stoploss
            if (rsi[i] > 70 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: momentum fading OR stoploss
            if (rsi[i] < 30 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: momentum pullback + trend alignment + volume + session
            long_setup = (rsi[i] < 30 and uptrend and vol_filter[i])
            short_setup = (rsi[i] > 70 and downtrend and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals