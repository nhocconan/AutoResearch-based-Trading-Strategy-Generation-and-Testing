#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and Volume Confirmation
Hypothesis: In trending markets (4h EMA50), RSI(14) pullbacks on 1h provide high-probability entries. Volume confirms momentum. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = False
    ema50_falling[0] = False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (already datetime64 index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 60  # For RSI and EMA50
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI mean reversion or stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) OR stoploss
            if (rsi[i] >= 70 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) OR stoploss
            if (rsi[i] <= 30 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI pullback + trend + volume
            rsi_pullback_long = rsi[i] < 40 and rsi[i] > rsi[i-1]  # bouncing from oversold
            rsi_pullback_short = rsi[i] > 60 and rsi[i] < rsi[i-1]  # declining from overbought
            
            bull_entry = rsi_pullback_long and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = rsi_pullback_short and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals