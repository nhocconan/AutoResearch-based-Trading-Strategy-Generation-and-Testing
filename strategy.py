#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_With_TimeFilter_v1
Hypothesis: Daily Camarilla pivot (S3/R3) breakouts on 4h chart filtered by 1h momentum and session.
Trades only during 08:00-20:00 UTC to avoid low-liquidity hours. Uses 4h for direction, 1h for timing.
Target: 15-30 trades/year per symbol with 0.20 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous close for calculation
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]
    
    # Daily range
    range_1d = high_1d - low_1d
    
    # S3 and R3 levels
    S3 = close_prev - (range_1d * 1.1000 / 4)  # Slightly tighter for more signals
    R3 = close_prev + (range_1d * 1.1000 / 4)
    
    # Align to 4h timeframe
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA20 trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h momentum filter: RSI(14) not extreme
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(S3_4h[i]) or np.isnan(R3_4h[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or
            not session_ok[i]):
            signals[i] = 0.0
            continue
        
        # Long: price above R3_4h, above 4h EMA20, RSI not overbought
        long_cond = (close[i] > R3_4h[i] and 
                    close[i] > ema_4h_aligned[i] and
                    rsi[i] < 70)
        
        # Short: price below S3_4h, below 4h EMA20, RSI not oversold
        short_cond = (close[i] < S3_4h[i] and 
                     close[i] < ema_4h_aligned[i] and
                     rsi[i] > 30)
        
        if long_cond and position != 1:
            position = 1
            signals[i] = position_size
        elif short_cond and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1d_Camarilla_Breakout_With_TimeFilter_v1"
timeframe = "1h"
leverage = 1.0