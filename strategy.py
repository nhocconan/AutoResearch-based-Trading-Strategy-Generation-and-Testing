#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d 50-period EMA for trend filter (long-term trend)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_12h[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_val) or 
            np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA50 and RSI > 50 (bullish momentum)
            if close_val > ema_val and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 and RSI < 50 (bearish momentum)
            elif close_val < ema_val and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA50 or ATR-based stop
            if close_val < ema_val or close_val < prices['high'].iloc[i] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA50 or ATR-based stop
            if close_val > ema_val or close_val > prices['low'].iloc[i] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1dEMA50_RSI_Momentum_V1
# Uses 1-day EMA50 as trend filter and 1-day RSI(14) for momentum confirmation
# Enters long when 12h price above EMA50 and RSI > 50
# Enters short when 12h price below EMA50 and RSI < 50
# Exits on EMA50 cross or 2*ATR stoploss (using 12h ATR)
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_1dEMA50_RSI_Momentum_V1"
timeframe = "12h"
leverage = 1.0