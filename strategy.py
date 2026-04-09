#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly close for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(prices), np.nan)  # Will fill with daily values, then align
    s4 = np.full(len(prices), np.nan)
    prev_high = np.full(len(prices), np.nan)
    prev_low = np.full(len(prices), np.nan)
    
    # We need daily OHLC to calculate pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivots for each daily bar
    r4_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    prev_high_1d = np.full(len(df_1d), np.nan)
    prev_low_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        r4_1d[i] = pc + (ph - pl) * 1.1 / 2
        s4_1d[i] = pc - (ph - pl) * 1.1 / 2
        prev_high_1d[i] = ph
        prev_low_1d[i] = pl
    
    # Align daily values to 1d timeframe (same frequency, just shifted for lookback)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or 
            np.isnan(prev_high_1d_aligned[i]) or 
            np.isnan(prev_low_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range OR weekly trend turns bearish
            if (close[i] <= prev_high_1d_aligned[i] and close[i] >= prev_low_1d_aligned[i]) or \
               (close[i] < weekly_ema_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range OR weekly trend turns bullish
            if (close[i] <= prev_high_1d_aligned[i] and close[i] >= prev_low_1d_aligned[i]) or \
               (close[i] > weekly_ema_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation AND weekly trend bullish
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > r4_1d_aligned[i] and 
                vol_ratio > 1.5 and 
                close[i] > weekly_ema_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation AND weekly trend bearish
            elif (close[i] < s4_1d_aligned[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < weekly_ema_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals