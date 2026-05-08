#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy using 4h Supertrend for direction and 1h RSI for entry timing, filtered by session (08-20 UTC).
# Long when 4h Supertrend is bullish AND 1h RSI < 30 (oversold) AND session active.
# Short when 4h Supertrend is bearish AND 1h RSI > 70 (overbought) AND session active.
# Exit when RSI crosses back to neutral (40 for long, 60 for short) or Supertrend flips.
# Uses lower position size (0.20) to control drawdown and session filter to reduce noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency to avoid fee drag.

name = "1h_Supertrend_RSI_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 4h
    tr4h = np.maximum(df_4h['high'] - df_4h['low'], 
                      np.maximum(np.abs(df_4h['high'] - np.roll(df_4h['close'], 1)),
                                 np.abs(df_4h['low'] - np.roll(df_4h['close'], 1))))
    tr4h[0] = df_4h['high'].iloc[0] - df_4h['low'].iloc[0]
    atr4h = pd.Series(tr4h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl4h = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl4h + (multiplier * atr4h)
    lower_band = hl4h - (multiplier * atr4h)
    
    # Initialize Supertrend
    supertrend = np.full_like(df_4h['close'], np.nan, dtype=float)
    direction = np.full_like(df_4h['close'], 1, dtype=int)  # 1 for up, -1 for down
    
    for i in range(1, len(df_4h)):
        if np.isnan(atr4h[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        if df_4h['close'].iloc[i] > upper_band[i-1]:
            direction[i] = 1
        elif df_4h['close'].iloc[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 1h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1h RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # RSI and session warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h Supertrend bullish, 1h RSI oversold, session active
            long_cond = (direction_aligned[i] == 1) and (rsi[i] < 30) and session_filter[i]
            # Short conditions: 4h Supertrend bearish, 1h RSI overbought, session active
            short_cond = (direction_aligned[i] == -1) and (rsi[i] > 70) and session_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses above 40 OR Supertrend turns bearish
            if rsi[i] > 40 or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses below 60 OR Supertrend turns bullish
            if rsi[i] < 60 or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals