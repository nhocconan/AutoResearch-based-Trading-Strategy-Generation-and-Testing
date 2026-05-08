#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla pivot levels, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # P = (H + L + C) / 3
    # R1 = P + (H - L) * 1.1 / 12
    # S1 = P - (H - L) * 1.1 / 12
    P = (high_1d + low_1d + close_1d) / 3
    R1 = P + (high_1d - low_1d) * 1.1 / 12
    S1 = P - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe (these are fixed for the day)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily average volume for volume confirmation
    avg_vol_1d = np.mean(volume_1d)
    avg_vol_aligned = np.full(n, avg_vol_1d)  # constant for the day
    
    # 4h volume spike (current volume > 1.5x daily average)
    vol_spike = volume > (avg_vol_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above EMA34, and volume spike
            long_cond = (close[i] > R1_aligned[i]) and (close[i] > ema34_aligned[i]) and vol_spike[i]
            # Short: Close breaks below S1, below EMA34, and volume spike
            short_cond = (close[i] < S1_aligned[i]) and (close[i] < ema34_aligned[i]) and vol_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below EMA34
            if close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above EMA34
            if close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance with higher probability of continuation when broken with volume. 
# EMA34 filter ensures we only trade in the direction of the daily trend. 
# Volume spike confirms institutional participation. 
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend). 
# Target: 20-50 trades per year (80-200 total over 4 years) to minimize fee decay.