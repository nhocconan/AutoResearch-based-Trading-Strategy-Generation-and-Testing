#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
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
    
    # Pivot support/resistance levels (R1/S1)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 12h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend direction from 1w EMA34
            uptrend = close[i] > ema34_1w_aligned[i]
            downtrend = close[i] < ema34_1w_aligned[i]
            
            if uptrend:
                # Long on break above R1 in uptrend with volume
                long_cond = (close[i] > r1_12h[i] and 
                            volume[i] > vol_ma20[i])
                if long_cond:
                    signals[i] = 0.25
                    position = 1
            elif downtrend:
                # Short on break below S1 in downtrend with volume
                short_cond = (close[i] < s1_12h[i] and 
                             volume[i] > vol_ma20[i])
                if short_cond:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close below S1 or trend reversal
            if close[i] < s1_12h[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above R1 or trend reversal
            if close[i] > r1_12h[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout strategy on 12h timeframe, filtered by 1w EMA34 trend and volume confirmation.
# In uptrends (price > 1w EMA34), go long on break above R1; in downtrends, go short on break below S1.
# Exits when price returns to S1/R1 or trend reverses. Uses 12h volume > 20-period average for confirmation.
# Designed to capture institutional interest at key pivot levels while avoiding false breakouts.
# Targets 50-150 trades over 4 years to minimize fee drag. Works on BTC/ETH via institutional pivot levels.