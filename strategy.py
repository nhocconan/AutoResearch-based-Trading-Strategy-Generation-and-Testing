#!/usr/bin/env python3
"""
1h_12h_1d_Trend_Momentum_v1
Hypothesis: In both bull and bear markets, strong momentum aligned with higher timeframe trends provides edge.
Uses 12h EMA50 for trend filter and 1d RSI(14) for momentum filter, with 1h price action for entry timing.
Target: 60-150 total trades over 4 years (15-37/year) by requiring confluence of 12h trend, 1d momentum, and 1h breakout.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 12h EMA50 for trend filter (loaded once) ===
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d RSI(14) for momentum filter (loaded once) ===
    df_1d = get_htf_data(prices, '1d')
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h price action: breakout above/below recent range ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: 12h uptrend (price > EMA50), 1d bullish momentum (RSI > 50), 1h breakout above recent high
            if (close[i] > ema_50_12h_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                high[i] > highest_high[i-1]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: 12h downtrend (price < EMA50), 1d bearish momentum (RSI < 50), 1h breakdown below recent low
            elif (close[i] < ema_50_12h_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  low[i] < lowest_low[i-1]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: reverse signal or momentum fade
        elif position == 1:
            # Exit long: 12h trend fails OR 1d momentum fades OR 1h breakdown
            if (close[i] < ema_50_12h_aligned[i] or 
                rsi_1d_aligned[i] < 40 or 
                low[i] < lowest_low[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 12h trend fails OR 1d momentum fades OR 1h breakout
            if (close[i] > ema_50_12h_aligned[i] or 
                rsi_1d_aligned[i] > 60 or 
                high[i] > highest_high[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_12h_1d_Trend_Momentum_v1"
timeframe = "1h"
leverage = 1.0