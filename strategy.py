# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakouts with 1-day trend filter and volume spike work across bull/bear markets.
# Uses Camarilla levels from prior day, 1-day EMA34 trend, and volume > 1.5x average.
# Entry only on breakouts aligned with daily trend. Exit on reversal to pivot point (PP).
# Target: 20-40 trades/year to minimize fee drag. Works in trends via breakouts, in ranges via mean reversion to PP.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            pp[i] = np.nan
            r1[i] = np.nan
            s1[i] = np.nan
        else:
            hlc = high_1d[i-1] + low_1d[i-1] + close_1d[i-1]
            pp[i] = hlc / 3.0
            rng = high_1d[i-1] - low_1d[i-1]
            r1[i] = close_1d[i-1] + rng * 1.1 / 12.0
            s1[i] = close_1d[i-1] - rng * 1.1 / 12.0
    
    # Calculate 1-day EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1-day indicators to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, EMA34, and volume MA20
    start_idx = max(1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above R1 with 1-day EMA34 uptrend and volume
            if (price > r1_aligned[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below S1 with 1-day EMA34 downtrend and volume
            elif (price < s1_aligned[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot point (mean reversion)
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above pivot point (mean reversion)
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0