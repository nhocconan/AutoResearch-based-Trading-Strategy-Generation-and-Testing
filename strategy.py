#!/usr/bin/env python3
"""
12h_1d_RSI_Reversal_v1
Hypothesis: Trade mean reversion at daily RSI extremes with 12h price action confirmation. 
Long when daily RSI < 30 and 12h close > 12h open (bullish candle). 
Short when daily RSI > 70 and 12h close < 12h open (bearish candle).
Uses volume confirmation (1.5x average) and exits when RSI returns to neutral zone (40-60).
Designed for low-frequency, high-conviction trades that work in both bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR RSI ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 12H INDICATORS ===
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(open_price[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Candle direction
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # RSI thresholds
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_neutral = (rsi_1d_aligned[i] >= 40) & (rsi_1d_aligned[i] <= 60)
        
        # Long: daily RSI oversold + bullish 12h candle + volume
        long_signal = rsi_oversold & bullish_candle & strong_volume
        
        # Short: daily RSI overbought + bearish 12h candle + volume
        short_signal = rsi_overbought & bearish_candle & strong_volume
        
        # Exit: RSI returns to neutral zone
        exit_long = position == 1 & rsi_neutral
        exit_short = position == -1 & rsi_neutral
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals