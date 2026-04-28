#!/usr/bin/env python3
"""
12h_RSI34_1dEMA21_TopBottom_Reversal
Hypothesis: RSI(34) on 12h identifies overbought/oversold conditions, while EMA(21) on 1d provides the primary trend filter. In bull markets, we buy oversold bounces above the rising 1d EMA21; in bear markets, we sell overbought rejections below the falling 1d EMA21. This mean-reversion-with-trend strategy avoids counter-trend trades and works in both regimes. Target: 20-30 trades/year per symbol to minimize fee drag while capturing meaningful reversals at key support/resistance levels.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 12h RSI(34) for overbought/oversold signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/34, adjust=False, min_periods=34).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for RSI to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        # Trend filter from 1d EMA21
        uptrend = close[i] > ema_21_1d_aligned[i]
        downtrend = close[i] < ema_21_1d_aligned[i]
        
        # Entry conditions: trade with the trend from extreme RSI levels
        long_entry = oversold and uptrend
        short_entry = overbought and downtrend
        
        # Exit conditions: RSI returns to neutral territory or trend changes
        long_exit = (rsi[i] > 50) or (not uptrend)
        short_exit = (rsi[i] < 50) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_RSI34_1dEMA21_TopBottom_Reversal"
timeframe = "12h"
leverage = 1.0