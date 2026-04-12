#!/usr/bin/env python3
"""
1d_1w_keltner_breakout
Hypothesis: Daily Keltner Channel breakout with weekly trend filter and volume confirmation.
Works in bull/bear: In trending markets, ride breakouts; in ranging markets, filter out false signals via weekly trend.
Uses volatility-based channels (ATR) to adapt to changing market conditions.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

name = "1d_1w_keltner_breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily ATR(20) for Keltner Channels
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily EMA20 for Keltner center
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels: 2.0 * ATR
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above upper Keltner with weekly uptrend and volume
        if (close[i] > upper[i] and close[i] > ema20_1w_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below lower Keltner with weekly downtrend and volume
        elif (close[i] < lower[i] and close[i] < ema20_1w_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to EMA20
        elif position == 1 and close[i] < ema20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals