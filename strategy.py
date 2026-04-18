#!/usr/bin/env python3
"""
4h_Turtle_Soup_Reversal_With_Volume_Confirmation
Hypothesis: False breakouts at prior session highs/lows (Turtle Soup) often reverse in mean-reverting markets.
Long when price breaks below prior 24h low then reverses above it with volume > 1.5x average.
Short when price breaks above prior 24h high then reverses below it with volume confirmation.
Uses 12h timeframe for context: only take longs if price > 12h EMA50, shorts if price < 12h EMA50.
Designed for 4H timeframe with ~25-40 trades/year to minimize fee drag and work in ranging/mean-reverting markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h timeframe for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 24h high/low from 12h data (two 12h bars)
    # We need to look back two completed 12h bars for the prior session range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # For each 4h bar, the prior 24h session high/low is the max/min of the two most recent completed 12h bars
    # We'll calculate this using rolling window on 12h data then align
    session_high_12h = pd.Series(high_12h).rolling(window=2, min_periods=2).max().values
    session_low_12h = pd.Series(low_12h).rolling(window=2, min_periods=2).min().values
    
    # Align to 4h timeframe (these represent the completed prior 24h session)
    session_high_aligned = align_htf_to_ltf(prices, df_12h, session_high_12h)
    session_low_aligned = align_htf_to_ltf(prices, df_12h, session_low_12h)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA and rolling window
    
    for i in range(start_idx, n):
        if (np.isnan(session_high_aligned[i]) or np.isnan(session_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        session_high = session_high_aligned[i]
        session_low = session_low_aligned[i]
        ema_trend = ema_12h_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Turtle Soup Long: price breaks below session low then reverses above it
            # Entry condition: price > session_low AND we were below session_low in the prior bar
            if i > 0 and low[i-1] <= session_low and price > session_low and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: price breaks above session high then reverses below it
            elif i > 0 and high[i-1] >= session_high and price < session_high and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price breaks below session low (failed reversal) or trend fails
            if price < session_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above session high (failed reversal) or trend fails
            if price > session_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Turtle_Soup_Reversal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0