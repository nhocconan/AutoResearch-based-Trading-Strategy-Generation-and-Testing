#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_Volume"
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
    
    # Daily Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1/12
    # S1 = Pivot - (H - L) * 1.1/12
    # Use previous day's values (shifted by 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan  # first value invalid
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Volume confirmation: volume > 1.5 * 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    # Weekly trend filter: EMA50 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for 20-day vol MA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + weekly uptrend
            if (price > r1[i] and 
                vol_filter[i] and 
                price > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 + volume confirmation + weekly downtrend
            elif (price < s1[i] and 
                  vol_filter[i] and 
                  price < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to pivot or weekly trend fails
            if (price < pivot[i] or 
                price < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or weekly trend fails
            if (price > pivot[i] or 
                price > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals