#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly True Range for volatility filter (weekly ATR)
    tr = np.full(len(df_w), np.nan)
    atr_w = np.full(len(df_w), np.nan)
    for i in range(1, len(df_w)):
        high_low = df_w['high'].iloc[i] - df_w['low'].iloc[i]
        high_close = np.abs(df_w['high'].iloc[i] - df_w['close'].iloc[i-1])
        low_close = np.abs(df_w['low'].iloc[i] - df_w['close'].iloc[i-1])
        tr[i] = max(high_low, high_close, low_close)
        if i >= 14:
            atr_w[i] = np.mean(tr[i-13:i+1])
    
    atr_w_aligned = align_htf_to_ltf(prices, df_w, atr_w)
    
    # Calculate daily OHLC arrays for pivot calculation
    daily_high = high.copy()
    daily_low = low.copy()
    daily_close = close.copy()
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    pp = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    
    for i in range(1, n):
        ph = daily_high[i-1]
        pl = daily_low[i-1]
        pc = daily_close[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4[i]) or 
            np.isnan(s4[i]) or 
            np.isnan(prev_high[i]) or 
            np.isnan(prev_low[i]) or 
            np.isnan(atr_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high[i] and close[i] >= prev_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high[i] and close[i] >= prev_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volatility filter
            # Enter short: price closes below S4 with volatility filter
            if (close[i] > r4[i] and 
                atr_w_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            elif (close[i] < s4[i] and 
                  atr_w_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
    
    return signals