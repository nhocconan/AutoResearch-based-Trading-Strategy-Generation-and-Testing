#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    
    # Initialize Supertrend
    st = np.zeros(len(close_4h))
    dir = np.ones(len(close_4h))  # 1 for uptrend, -1 for downtrend
    st[0] = upper[0]
    dir[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > st[i-1]:
            dir[i] = 1
        elif close_4h[i] < st[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
        
        if dir[i] == 1:
            st[i] = max(lower[i], st[i-1])
        else:
            st[i] = min(upper[i], st[i-1])
    
    st_aligned = align_htf_to_ltf(prices, df_4h, st)
    dir_aligned = align_htf_to_ltf(prices, df_4h, dir)
    
    # Calculate 1d RSI(14) for entry timing
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        st_val = st_aligned[i]
        dir_val = dir_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_val = volume_1d[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(st_val) or np.isnan(dir_val) or 
            np.isnan(rsi_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (dir=1), RSI < 30 (oversold), volume above average
            if dir_val == 1 and rsi_val < 30 and vol_val > vol_avg_val:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (dir=-1), RSI > 70 (overbought), volume above average
            elif dir_val == -1 and rsi_val > 70 and vol_val > vol_avg_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: downtrend (dir=-1) or RSI > 70 (overbought)
            if dir_val == -1 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: uptrend (dir=1) or RSI < 30 (oversold)
            if dir_val == 1 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# 1h_4hSupertrend_1dRSI_Volume_Session
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction
# Enters long when 4h uptrend, 1d RSI < 30, and volume above average
# Enters short when 4h downtrend, 1d RSI > 70, and volume above average
# Exits when 4h trend reverses or RSI reaches opposite extreme
# Session filter: 08-20 UTC to avoid low-volume periods
# Designed for 1h timeframe with ~15-35 trades/year
name = "1h_4hSupertrend_1dRSI_Volume_Session"
timeframe = "1h"
leverage = 1.0