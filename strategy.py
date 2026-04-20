#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 14-period ATR on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR as EMA of TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
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
        atr_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volatility filter (ATR > 0.5% of price)
            if rsi_val < 30 and atr_val > 0.005 * close_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volatility filter
            elif rsi_val > 70 and atr_val > 0.005 * close_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_RSI14_1d_VolatilityFilter_Session_v1
# Uses 1d 14-period RSI for mean reversion signals
# Requires 1d ATR > 0.5% of price for volatility filter (avoids low-vol chop)
# Session filter: 8-20 UTC to focus on active trading hours
# Mean reversion: long when RSI<30, short when RSI>70, exit at RSI=50
# Designed for 12h timeframe with ~15-30 trades/year
name = "12h_RSI14_1d_VolatilityFilter_Session_v1"
timeframe = "12h"
leverage = 1.0