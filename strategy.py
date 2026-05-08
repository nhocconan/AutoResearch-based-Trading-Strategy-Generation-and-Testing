#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's pivot points (HLC/3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Pivot support/resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 4h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: 1d EMA34
            uptrend = close[i] > ema34_1d_aligned[i]
            downtrend = close[i] < ema34_1d_aligned[i]
            
            if uptrend:
                # Uptrend: long on break above R1
                long_cond = (close[i] > r1_4h[i] and 
                            volume[i] > vol_ma20[i])
            elif downtrend:
                # Downtrend: short on break below S1
                short_cond = (close[i] < s1_4h[i] and 
                             volume[i] > vol_ma20[i])
            else:
                # Around EMA: no trades
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S1 or trend reversal
            exit_cond = (close[i] < s1_4h[i] or 
                        close[i] < ema34_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R1 or trend reversal
            exit_cond = (close[i] > r1_4h[i] or 
                        close[i] > ema34_1d_aligned[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout strategy with 1d EMA34 trend filter and volume confirmation.
# In uptrend (price > 1d EMA34): long on break above R1, exit on break below S1 or trend reversal.
# In downtrend (price < 1d EMA34): short on break below S1, exit on break above R1 or trend reversal.
# Uses volume confirmation to avoid false breakouts. Designed for 4h timeframe with institutional
# pivot levels that work in both bull and bear markets. Targets 50-150 trades over 4 years
# (12-37/year) to minimize fee drag. Uses discrete sizing (0.25) to reduce churn.