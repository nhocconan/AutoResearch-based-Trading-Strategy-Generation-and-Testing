#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_1wTrend
Hypothesis: Daily KAMA direction filter with RSI mean-reversion and weekly trend alignment. Uses Choppiness Index to filter ranging markets. Targets 10-20 trades/year by requiring KAMA trend confirmation, RSI extremes, and weekly trend alignment. Works in bull/bear via trend filter and mean-reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(np.sum(atr, axis=0) / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # Align weekly trend
    trend_up = close > ema_20_1w_aligned
    trend_down = close < ema_20_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # Entry conditions
        # Long: price above KAMA + RSI oversold (30) + weekly uptrend + not choppy (CHOP < 61.8)
        long_entry = (kama_up[i] and 
                     rsi[i] < 30 and 
                     trend_up[i] and 
                     chop[i] < 61.8)
        
        # Short: price below KAMA + RSI overbought (70) + weekly downtrend + not choppy (CHOP < 61.8)
        short_entry = (kama_down[i] and 
                      rsi[i] > 70 and 
                      trend_down[i] and 
                      chop[i] < 61.8)
        
        # Exit conditions
        # Exit long: RSI overbought (70) or weekly trend changes
        long_exit = (rsi[i] > 70 or 
                    (position == 1 and not trend_up[i]))
        
        # Exit short: RSI oversold (30) or weekly trend changes
        short_exit = (rsi[i] < 30 or 
                     (position == -1 and not trend_down[i]))
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_1wTrend"
timeframe = "1d"
leverage = 1.0