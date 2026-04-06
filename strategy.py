#!/usr/bin/env python3
"""
6h Bollinger Band Breakout with Weekly Trend Filter
Hypothesis: In trending markets, breakouts beyond 2.0 Bollinger Bands on 6h capture momentum when aligned with weekly trend.
In ranging markets, mean reversion at band edges with RSI filter prevents false breakouts.
Works in bull (long when price > upper band in uptrend) and bear (short when price < lower band in downtrend).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_breakout_weekly_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2.0 * std_20)
    lower_band = sma_20 - (2.0 * std_20)
    
    # 6h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For BBands and weekly EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(sma_20[i]) or
            np.isnan(std_20[i]) or np.isnan(rsi[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = ema_50_weekly_aligned[i] > close[i]
        weekly_downtrend = ema_50_weekly_aligned[i] < close[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: mean reversion OR stoploss
            if (close[i] <= sma_20[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: mean reversion OR stoploss
            if (close[i] >= sma_20[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout with weekly trend filter
            long_breakout = (close[i] > upper_band[i] and weekly_uptrend and rsi[i] > 50)
            short_breakout = (close[i] < lower_band[i] and weekly_downtrend and rsi[i] < 50)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals