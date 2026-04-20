# !/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 50-period SMA on 1d close
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        sma_val = sma_50_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(sma_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above SMA (bullish bias)
            if rsi_val < 30 and close_val > sma_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price below SMA (bearish bias)
            elif rsi_val > 70 and close_val < sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price crosses below SMA
            if rsi_val > 70 or close_val < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price crosses above SMA
            if rsi_val < 30 or close_val > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_1d_RSI30_70_SMA50_Session_v1
# Uses 1d RSI(14) for overbought/oversold signals
# Uses 1d SMA(50) for trend bias
# Entry: RSI < 30 + price > SMA50 (long) OR RSI > 70 + price < SMA50 (short)
# Exit: RSI > 70 or price < SMA50 (long) OR RSI < 30 or price > SMA50 (short)
# Session filter: 8-20 UTC to focus on active trading hours
# Position size: 0.25 (25% of capital)
# Designed for 6h timeframe with ~15-30 trades/year
name = "6h_1d_RSI30_70_SMA50_Session_v1"
timeframe = "6h"
leverage = 1.0