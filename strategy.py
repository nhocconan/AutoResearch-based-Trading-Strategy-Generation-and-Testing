#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4-hour and 1-day timeframes for signal direction, with 1h for entry timing.
Uses 4h Supertrend for trend direction, 1d RSI for momentum filter, and 1h price action for entry.
In uptrend (4h Supertrend up) with bullish momentum (1d RSI > 50), buy on 1h pullbacks to EMA21.
In downtrend (4h Supertrend down) with bearish momentum (1d RSI < 50), sell on 1h bounces to EMA21.
Designed for 15-37 trades/year (60-150 total) to minimize fee drift while capturing trend continuations.
Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + 3.0 * atr_4h
    lower_band = hl2 - 3.0 * atr_4h
    
    # Supertrend calculation
    supertrend = np.full_like(close_4h, np.nan, dtype=float)
    direction = np.ones_like(close_4h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_4h[i-1]) or np.isnan(close_4h[i-1]):
            supertrend[i] = supertrend[i-1] if i > 0 else np.nan
            direction[i] = direction[i-1] if i > 0 else 1
            continue
            
        # Update bands
        if close_4h[i-1] > supertrend[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close_4h[i-1] < supertrend[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Determine trend
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 4h Supertrend direction to 1h
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction.astype(float))
    
    # Align 1d RSI to 1h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h EMA21 for entry timing
    close_1h = prices['close'].values
    ema_21 = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_1h[i]
        st_direction = supertrend_direction_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        ema_val = ema_21[i]
        
        if position == 0:
            # Enter long: 4h uptrend, 1d RSI > 50, price pulls back to EMA21
            if (st_direction == 1 and 
                rsi_val > 50 and 
                price_close >= ema_val * 0.998 and  # Allow small tolerance
                price_close <= ema_val * 1.002):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend, 1d RSI < 50, price bounces to EMA21
            elif (st_direction == -1 and 
                  rsi_val < 50 and 
                  price_close >= ema_val * 0.998 and
                  price_close <= ema_val * 1.002):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend reversal on 4h Supertrend
            exit_signal = False
            
            if position == 1 and st_direction == -1:
                exit_signal = True
            elif position == -1 and st_direction == 1:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Supertrend4h_RSI1d_EMA21_Session"
timeframe = "1h"
leverage = 1.0