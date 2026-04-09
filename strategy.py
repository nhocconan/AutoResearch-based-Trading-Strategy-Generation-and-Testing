#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend
    close_w = df_w['close'].values
    ema_w = np.full(len(df_w), np.nan)
    if len(close_w) >= 21:
        alpha = 2.0 / (21 + 1)
        ema_w[20] = np.mean(close_w[:21])
        for i in range(21, len(close_w)):
            ema_w[i] = alpha * close_w[i] + (1 - alpha) * ema_w[i-1]
    
    # Align weekly EMA to daily
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    
    for i in range(1, n):
        ph = float(high[i-1])
        pl = float(low[i-1])
        pc = float(close[i-1])
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Volume confirmation: 3-period average (3-day)
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4[i]) or 
            np.isnan(s4[i]) or 
            np.isnan(prev_high[i]) or 
            np.isnan(prev_low[i]) or 
            np.isnan(vol_ma_3[i]) or 
            np.isnan(ema_w_aligned[i])):
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
            # Enter long: price closes above R4 with volume confirmation and weekly uptrend
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] > r4[i] and 
                vol_ratio > 2.0 and 
                close[i] > ema_w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation and weekly downtrend
            elif (close[i] < s4[i] and 
                  vol_ratio > 2.0 and 
                  close[i] < ema_w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals