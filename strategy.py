#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Hypothesis: 12-hour breakout of daily Camarilla pivot levels with 1-day EMA50 trend filter and volume confirmation.
# Long when price breaks above R4 resistance with price > daily EMA50 and volume > 1.5x 24-bar average.
# Short when price breaks below S4 support with price < daily EMA50 and volume > 1.5x 24-bar average.
# Exit when price returns to opposite pivot level (S4 for longs, R4 for shorts).
# Daily Camarilla levels calculated from prior day's OHLC: PP=(H+L+C)/3, R4=C+(H-L)*1.1/2, S4=C-(H-L)*1.1/2.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    pp = np.full(len(df_d), np.nan)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):  # Start from 1 to use previous day
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
    
    # Align daily Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    
    # Load daily data ONCE before loop for EMA50
    close_d = df_d['close'].values
    ema_50_d = np.full(len(close_d), np.nan)
    if len(close_d) >= 50:
        # Initialize EMA with SMA of first 50 periods
        sma = np.mean(close_d[:50])
        ema_50_d[49] = sma
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_d)):
            ema_50_d[i] = (close_d[i] - ema_50_d[i-1]) * multiplier + ema_50_d[i-1]
    
    # Align daily EMA50 to 12h timeframe
    ema_50_d_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # Volume confirmation: 24-period average (24*12h = 12 days)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below S4 level
            if close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above R4 level
            if close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R4 with trend and volume filters
            if (close[i] > r4_aligned[i] and 
                close[i] > ema_50_d_aligned[i] and 
                volume[i] > vol_ma_24[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with trend and volume filters
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema_50_d_aligned[i] and 
                  volume[i] > vol_ma_24[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals