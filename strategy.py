#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend with 1w Williams %R filter + volume confirmation
    # Long: KAMA rising AND Williams %R < -80 (oversold) AND volume > 1.5x 20-day average
    # Short: KAMA falling AND Williams %R > -20 (overbought) AND volume > 1.5x 20-day average
    # Exit: KAMA reverses direction OR Williams %R crosses midline (-50)
    # Using 1d timeframe for lower trade frequency (target 7-25/year), weekly Williams %R for extreme sentiment,
    # and volume confirmation to avoid false signals. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.full(n, np.nan)
    for i in range(10, n):  # min_periods=10 for ER
        if i >= 10:
            direction = np.abs(close[i] - close[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
    
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = np.full(n, np.nan)
    for i in range(10, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fastest - slowest) + slowest) ** 2
        else:
            sc[i] = 0
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction (1 = rising, -1 = falling, 0 = flat)
    kama_dir = np.zeros(n)
    for i in range(1, n):
        if kama[i] > kama[i-1]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # Get weekly data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    williams_r = np.full(len(close_1w), np.nan)
    for i in range(13, len(close_1w)):  # min_periods=14
        highest_high = np.max(high_1w[i-13:i+1])
        lowest_low = np.min(low_1w[i-13:i+1])
        if highest_high - lowest_low != 0:
            williams_r[i] = (highest_high - close_1w[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Align weekly Williams %R to 1d
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Get 1d volume for confirmation (>1.5x 20-day average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_dir[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (kama_dir[i] == 1 and 
                     williams_r_aligned[i] < -80 and 
                     volume_spike[i])
        short_entry = (kama_dir[i] == -1 and 
                      williams_r_aligned[i] > -20 and 
                      volume_spike[i])
        
        # Exit conditions: KAMA reverses OR Williams %R crosses midline (-50)
        long_exit = (kama_dir[i] == -1) or (williams_r_aligned[i] > -50)
        short_exit = (kama_dir[i] == 1) or (williams_r_aligned[i] < -50)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_williamsr_volume_v1"
timeframe = "1d"
leverage = 1.0