#!/usr/bin/env python3
"""
1d_Fibonacci_Retracement_Breakout_WeeklyTrend
Hypothesis: On the daily timeframe, price retracements to key Fibonacci levels (38.2%, 61.8%) of the weekly swing
combined with weekly trend alignment and volume confirmation capture high-probability continuation moves.
Works in bull/bear by following the weekly trend direction. Target: 10-25 trades/year (40-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for swing and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly swing high/low over last 4 weeks
    swing_high = pd.Series(df_1w['high']).rolling(window=4, min_periods=4).max().shift(1).values
    swing_low = pd.Series(df_1w['low']).rolling(window=4, min_periods=4).min().shift(1).values
    swing_range = swing_high - swing_low
    
    # Fibonacci retracement levels: 38.2% and 61.8%
    fib_382 = swing_low + swing_range * 0.382
    fib_618 = swing_low + swing_range * 0.618
    
    # Align to daily timeframe
    fib_382_d = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_618_d = align_htf_to_ltf(prices, df_1w, fib_618)
    swing_high_d = align_htf_to_ltf(prices, df_1w, swing_high)
    swing_low_d = align_htf_to_ltf(prices, df_1w, swing_low)
    
    # Weekly trend: 8-period EMA of weekly close
    ema_8w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8w_d = align_htf_to_ltf(prices, df_1w, ema_8w)
    
    # Daily volume filter: >1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(fib_382_d[i]) or np.isnan(fib_618_d[i]) or
            np.isnan(swing_high_d[i]) or np.isnan(swing_low_d[i]) or
            np.isnan(ema_8w_d[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        weekly_trend_up = ema_8w_d[i] > close[i-7] if i >= 7 else ema_8w_d[i] > close[0]
        weekly_trend_down = ema_8w_d[i] < close[i-7] if i >= 7 else ema_8w_d[i] < close[0]
        
        if position == 0:
            # Long: pullback to 61.8% in uptrend with volume
            if (price >= fib_618_d[i] * 0.995 and price <= fib_618_d[i] * 1.005) and \
               weekly_trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: pullback to 38.2% in downtrend with volume
            elif (price >= fib_382_d[i] * 0.995 and price <= fib_382_d[i] * 1.005) and \
                 weekly_trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly trend reverses or price reaches swing high
            if not weekly_trend_up or price >= swing_high_d[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend reverses or price reaches swing low
            if not weekly_trend_down or price <= swing_low_d[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Fibonacci_Retracement_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0