#!/usr/bin/env python3
"""
1d_1w_Keltner_Breakout_Trend_v1
Hypothesis: Uses weekly ATR-based Keltner channels with daily trend filter to capture strong trends in both bull and bear markets. 
Keltner breakouts with trend alignment reduce whipsaws, while weekly timeframe ensures low trade frequency (target: 8-12 trades/year).
Works in bull markets by catching breakouts and in bear markets by avoiding false signals during consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for Keltner channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR(10) for Keltner channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR undefined
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA(20) for middle line
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: Upper = EMA + 2*ATR, Lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # Align Keltner channels to daily timeframe (wait for weekly close)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Daily EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > keltner_upper_aligned[i]  # Break above upper channel
        breakdown_down = close[i] < keltner_lower_aligned[i]  # Break below lower channel
        
        # Trend filter: price relative to daily EMA50
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Entry conditions: only trade breakouts in direction of trend
        long_entry = breakout_up and uptrend
        short_entry = breakdown_down and downtrend
        
        # Exit conditions: return to middle line or trend reversal
        keltner_middle = ema_20  # Will be aligned below
        # We need to align the middle line as well
        keltner_middle_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
        long_exit = close[i] < keltner_middle_aligned[i]  # Return below middle
        short_exit = close[i] > keltner_middle_aligned[i]  # Return above middle
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals