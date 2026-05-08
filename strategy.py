#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Fibonacci_Retracement_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h ATR(14) for swing calculation
    high_pd = pd.Series(high)
    low_pd = pd.Series(low)
    close_pd = pd.Series(close)
    tr1 = high_pd - low_pd
    tr2 = abs(high_pd - close_pd.shift(1))
    tr3 = abs(low_pd - close_pd.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Identify swing highs and lows using 4-period lookback
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(4, n-4):
        if high[i] == np.max(high[i-4:i+5]):
            swing_high[i] = True
        if low[i] == np.min(low[i-4:i+5]):
            swing_low[i] = True
    
    # Find most recent swing points for Fibonacci calculation
    last_swing_high_idx = np.where(swing_high)[0]
    last_swing_low_idx = np.where(swing_low)[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            len(last_swing_high_idx) == 0 or len(last_swing_low_idx) == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get most recent swing points before current bar
        valid_highs = last_swing_high_idx[last_swing_high_idx < i]
        valid_lows = last_swing_low_idx[last_swing_low_idx < i]
        
        if len(valid_highs) == 0 or len(valid_lows) == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        recent_high_idx = valid_highs[-1]
        recent_low_idx = valid_lows[-1]
        
        swing_high_price = high[recent_high_idx]
        swing_low_price = low[recent_low_idx]
        
        # Skip if swing points are too close (invalid range)
        if swing_high_price <= swing_low_price:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Fibonacci levels
        swing_range = swing_high_price - swing_low_price
        fib_382 = swing_low_price + 0.382 * swing_range
        fib_618 = swing_low_price + 0.618 * swing_range
        
        # Trend filter: 12h EMA50 slope
        uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
        downtrend = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
        
        if position == 0:
            # Long: pullback to 61.8% in uptrend, break above 38.2%
            long_cond = (uptrend and 
                        low[i] <= fib_618 * 1.005 and  # Allow small tolerance
                        high[i] > fib_382 and
                        close[i] > fib_382)
            
            # Short: pullback to 38.2% in downtrend, break below 61.8%
            short_cond = (downtrend and 
                         high[i] >= fib_382 * 0.995 and  # Allow small tolerance
                         low[i] < fib_618 and
                         close[i] < fib_618)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 61.8% or trend changes
            if close[i] < fib_618 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 38.2% or trend changes
            if close[i] > fib_382 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals