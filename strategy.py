#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Range_200MA_v1
Hypothesis: Uses KAMA to determine trend direction, RSI for overbought/oversold conditions, and 200-day MA as a long-term filter. Trades only in the direction of the long-term trend with momentum confirmation. Designed for low trade frequency (<25/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate KAMA (adaptive moving average) for daily trend
    # Efficiency ratio: price change over 10 periods / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # 200-day moving average (long-term filter)
    ma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 200  # MA200 needs 200 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ma200[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        ma200_val = ma200[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long conditions: price above KAMA (uptrend), RSI not overbought, above 200MA, weekly uptrend, volume
            if (close_val > kama_val and 
                rsi_val < 70 and 
                close_val > ma200_val and 
                close_val > ema50_1w_val and 
                vol_conf):
                signals[i] = size
                position = 1
            # Short conditions: price below KAMA (downtrend), RSI not oversold, below 200MA, weekly downtrend, volume
            elif (close_val < kama_val and 
                  rsi_val > 30 and 
                  close_val < ma200_val and 
                  close_val < ema50_1w_val and 
                  vol_conf):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_Range_200MA_v1"
timeframe = "1d"
leverage = 1.0