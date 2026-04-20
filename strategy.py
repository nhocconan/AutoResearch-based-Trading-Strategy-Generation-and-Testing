#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 1d trend filter (EMA50) and 4h momentum (RSI14)
# Entry: Price > 1d EMA50 AND RSI14(4h) > 50 for long; opposite for short
# Exit: Opposite condition or time-based (max 24h hold) to reduce churn
# Session filter: 08-20 UTC to avoid low-volume hours
# Position size: 0.20 (20%) to manage drawdown in volatile markets
# Target: 15-30 trades/year by requiring multiple confluence factors

name = "1h_1d_EMA50_4h_RSI14_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 4h data for momentum (RSI14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI14
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Session filter: 8-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        bars_since_entry += 1
        
        # Force exit after 24 bars (24h) to prevent stale positions
        if bars_since_entry >= 24 and position != 0:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(ema_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Long: Above 1d EMA50 AND 4h RSI > 50 (bullish momentum)
            if close_val > ema_val and rsi_val > 50:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: Below 1d EMA50 AND 4h RSI < 50 (bearish momentum)
            elif close_val < ema_val and rsi_val < 50:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Long exit: Below EMA50 OR RSI < 40 (loss of momentum)
            if close_val < ema_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Above EMA50 OR RSI > 60 (loss of momentum)
            if close_val > ema_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals