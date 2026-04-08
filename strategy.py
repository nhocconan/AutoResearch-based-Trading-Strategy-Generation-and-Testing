#!/usr/bin/env python3
"""
1h_4h_1d_rsi_momentum_v1
Hypothesis: Combine 1d RSI momentum (trend filter) with 4h RSI overbought/oversold 
and 1h entry timing. In bull markets, 1d RSI > 50 indicates uptrend; in bear markets, 
1d RSI < 50 indicates downtrend. Enter on 4h RSI extremes with 1h price action 
confirmation. Volume filter reduces false signals. Target: 15-30 trades/year.
Timeframe: 1h (primary), HTF: 4h (RSI), 1d (trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (RSI > 50 = uptrend, < 50 = downtrend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 4h data for entry signal (RSI extremes)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                # Hold position outside session
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h RSI returns to neutral (50) or 1d trend changes
            if rsi_4h_aligned[i] >= 50 or rsi_1d_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: 4h RSI returns to neutral (50) or 1d trend changes
            if rsi_4h_aligned[i] <= 50 or rsi_1d_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 4h RSI oversold (<30) + 1d uptrend (>50) + volume
            if (rsi_4h_aligned[i] < 30 and rsi_1d_aligned[i] > 50 and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h RSI overbought (>70) + 1d downtrend (<50) + volume
            elif (rsi_4h_aligned[i] > 70 and rsi_1d_aligned[i] < 50 and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.20
    
    return signals