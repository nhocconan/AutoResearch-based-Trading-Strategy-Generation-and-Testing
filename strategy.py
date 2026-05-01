#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and session filter (08-20 UTC).
# Uses 4h Camarilla pivot levels (R1/S1) for institutional breakout detection.
# Filtered by 4h EMA20 trend and session filter to reduce noise and avoid overtrading.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Discrete position sizing (0.20) balances return and drawdown.

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h Camarilla pivot levels (R1, S1)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3
    rang = high_4h - low_4h
    r1 = pivot + rang * 1.1 / 4
    s1 = pivot - rang * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA20
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        
        # Trend filter: 4h EMA20 direction
        uptrend = curr_close > ema_20_4h_aligned[i]
        downtrend = curr_close < ema_20_4h_aligned[i]
        
        # Camarilla breakout conditions (R1/S1 = institutional breakout levels)
        breakout_up = curr_close > r1_aligned[i]   # break above R1
        breakout_down = curr_close < s1_aligned[i]  # break below S1
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R1 AND uptrend
            if breakout_up and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below S1 AND downtrend
            elif breakout_down and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S1 (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on breakout above R1 (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals