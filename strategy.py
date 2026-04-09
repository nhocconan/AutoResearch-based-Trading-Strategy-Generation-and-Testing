#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    pp = np.full(len(df_d), np.nan)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 10-period average (10*12h = 120h ~ 5 days)
    vol_ma_10 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 10:
            vol_sum -= volume[i-10]
        if i >= 9:
            vol_ma_10[i] = vol_sum / 10
    
    # Momentum filter: 12-period RSI to avoid counter-trend entries
    rsi = np.full(n, np.nan)
    change = np.diff(close, prepend=close[0])
    up = np.where(change > 0, change, 0)
    down = np.where(change < 0, -change, 0)
    gain = np.full(n, np.nan)
    loss = np.full(n, np.nan)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            gain[i] = up[i]
            loss[i] = down[i]
        else:
            gain[i] = up[i]
            loss[i] = down[i]
        
        if i >= 12:
            if i == 12:
                avg_gain[i] = np.mean(gain[i-11:i+1])
                avg_loss[i] = np.mean(loss[i-11:i+1])
            else:
                avg_gain[i] = (avg_gain[i-1] * 11 + gain[i]) / 12
                avg_loss[i] = (avg_loss[i-1] * 11 + loss[i]) / 12
            
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_10[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation and bullish momentum
            if (close[i] > r4_aligned[i] and 
                volume[i] > vol_ma_10[i] * 1.5 and
                rsi[i] > 50):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation and bearish momentum
            elif (close[i] < s4_aligned[i] and 
                  volume[i] > vol_ma_10[i] * 1.5 and
                  rsi[i] < 50):
                position = -1
                signals[i] = -0.25
    
    return signals