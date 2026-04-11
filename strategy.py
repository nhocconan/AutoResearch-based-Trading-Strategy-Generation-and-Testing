#!/usr/bin/env python3
"""
6h_1w_elliott_wave_fib
Strategy: 6h Elliott Wave-inspired trend continuation with 1-week Fibonacci retracement levels
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 1-week Fibonacci retracement levels (38.2%, 61.8%) from the prior weekly swing to identify institutional support/resistance. Enters on 6h pullbacks to these levels in the direction of the higher timeframe trend (1-week Supertrend). Designed for low frequency (15-30 trades/year) to minimize fee decay while capturing high-probability trend continuations in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_elliott_wave_fib"
timeframe = "6h"
leverage = 1.0

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    # Upper and Lower Bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if np.isnan(atr[i-1]) or np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]):
            continue
            
        # Upper band logic
        if close[i-1] > upperband[i-1]:
            upperband[i] = upperband[i-1]
        else:
            upperband[i] = hl2[i] + (multiplier * atr[i])
            
        # Lower band logic
        if close[i-1] < lowerband[i-1]:
            lowerband[i] = lowerband[i-1]
        else:
            lowerband[i] = hl2[i] - (multiplier * atr[i])
            
        # Trend direction
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        # Supertrend value
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
            
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1-week Swing Points for Fibonacci ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Find weekly swing high and low (using 5-period window)
    def find_swing_points(arr, window=5):
        """Find swing highs and lows"""
        highs = np.full_like(arr, np.nan)
        lows = np.full_like(arr, np.nan)
        
        for i in range(window, len(arr) - window):
            # Swing high: highest high in window
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = arr[i]
            # Swing low: lowest low in window
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = arr[i]
        return highs, lows
    
    swing_high_1w, swing_low_1w = find_swing_points(high_1w, 5)
    
    # Find most recent swing high and low
    def get_most_recent_swing(values):
        """Get the most recent non-NaN swing value"""
        for i in range(len(values)-1, -1, -1):
            if not np.isnan(values[i]):
                return values[i]
        return np.nan
    
    # Calculate Fibonacci levels from weekly swing
    def calculate_fib_levels(high_val, low_val):
        """Calculate 38.2% and 61.8% Fibonacci retracement levels"""
        if np.isnan(high_val) or np.isnan(low_val) or high_val <= low_val:
            return np.nan, np.nan
        diff = high_val - low_val
        fib_382 = high_val - (diff * 0.382)
        fib_618 = high_val - (diff * 0.618)
        return fib_382, fib_618
    
    # Get most recent swing points
    recent_swing_high = get_most_recent_swing(swing_high_1w)
    recent_swing_low = get_most_recent_swing(swing_low_1w)
    
    # Calculate Fibonacci levels
    fib_382, fib_618 = calculate_fib_levels(recent_swing_high, recent_swing_low)
    
    # Align Fibonacci levels to 6h timeframe (hold until new swing forms)
    fib_382_arr = np.full(len(prices), fib_382)
    fib_618_arr = np.full(len(prices), fib_618)
    
    # === 1-week Supertrend (trend filter) ===
    supertrend_1w, trend_1w = calculate_supertrend(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values,
        atr_period=10,
        multiplier=3.0
    )
    
    # Align Supertrend and trend direction to 6h
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    # === 6h RSI (entry timing) ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h Volume Filter ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(fib_382_arr[i]) or np.isnan(fib_618_arr[i]) or
            np.isnan(supertrend_1w_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Trend direction from 1-week Supertrend
        uptrend = trend_1w_aligned[i] > 0
        downtrend = trend_1w_aligned[i] < 0
        
        # Price near Fibonacci levels (within 0.5% tolerance)
        near_fib_382 = abs(price_close - fib_382_arr[i]) / fib_382_arr[i] < 0.005
        near_fib_618 = abs(price_close - fib_618_arr[i]) / fib_618_arr[i] < 0.005
        
        # Volume confirmation: above average volume
        volume_ok = volume[i] > vol_ma_20[i]
        
        # RSI conditions for pullback entry
        rsi_not_overbought = rsi[i] < 60
        rsi_not_oversold = rsi[i] > 40
        
        # Long conditions: uptrend + pullback to 61.8% Fib + volume + RSI not overbought
        long_signal = (
            uptrend and 
            near_fib_618 and 
            volume_ok and 
            rsi_not_overbought
        )
        
        # Short conditions: downtrend + pullback to 38.2% Fib + volume + RSI not oversold
        short_signal = (
            downtrend and 
            near_fib_382 and 
            volume_ok and 
            rsi_not_oversold
        )
        
        # Exit conditions: trend reversal or price moves against position
        exit_long = position == 1 and (not uptrend or price_close < supertrend_1w_aligned[i])
        exit_short = position == -1 and (not downtrend or price_close > supertrend_1w_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals