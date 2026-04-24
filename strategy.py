#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session timing.
- Primary timeframe: 1h for execution, HTF: 4h for trend direction.
- In 4h uptrend (close > EMA50): look for 1h RSI(2) < 10 for long entries.
- In 4h downtrend (close < EMA50): look for 1h RSI(2) > 90 for short entries.
- Only trade during active session (08-20 UTC) to avoid low-liquidity hours.
- Volume confirmation: current 1h volume > 1.5 * 20-period volume MA.
- Exit: RSI(2) crosses back above 60 (long) or below 40 (short) OR opposing signal.
- Discrete signal size: 0.20 to control drawdown and minimize fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Session filter: 08-20 UTC (already datetime64 index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 2)  # Need EMA50 and RSI(2) warmup
    
    for i in range(start_idx, n):
        # Skip if not in active session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_4h_aligned[i]
        curr_close = close[i]
        curr_rsi = rsi[i]
        prev_rsi = rsi[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i] and in_session[i]:
                if curr_close > ema50:  # 4h uptrend: look for long at RSI(2) < 10
                    if curr_rsi < 10:
                        signals[i] = 0.20
                        position = 1
                elif curr_close < ema50:  # 4h downtrend: look for short at RSI(2) > 90
                    if curr_rsi > 90:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: RSI(2) > 60 or 4h trend turns down
            if curr_rsi > 60 or curr_close < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI(2) < 40 or 4h trend turns up
            if curr_rsi < 40 or curr_close > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_4hEMA50Trend_Session_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0