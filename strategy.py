#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation
# Uses 1h timeframe with tight entry conditions to balance trade frequency and capture meaningful moves
# Camarilla levels from 4h provide clear structure for breakouts
# Breakout at R1/S1 with volume spike confirms participation
# 4h EMA50 trend filter ensures alignment with intermediate trend
# Session filter (08-20 UTC) reduces noise during low-activity periods
# Discrete position sizing: 0.20 (20% of capital) to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R1, S1, R2, S2
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = Close + Range * 1.1/12
    # S1 = Close - Range * 1.1/12
    # R2 = Close + Range * 1.1/6
    # S2 = Close - Range * 1.1/6
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r1_4h = close_4h + range_4h * 1.1 / 12.0
    s1_4h = close_4h - range_4h * 1.1 / 12.0
    r2_4h = close_4h + range_4h * 1.1 / 6.0
    s2_4h = close_4h - range_4h * 1.1 / 6.0
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.8x 24-period average (4h equivalent)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r2_4h_aligned[i]) or np.isnan(s2_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R1 with volume spike AND price > 4h EMA50 (bullish trend)
            if (close[i] > r1_4h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 with volume spike AND price < 4h EMA50 (bearish trend)
            elif (close[i] < s1_4h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S1 OR below 4h EMA50 (trend change)
            if close[i] < s1_4h_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above R1 OR above 4h EMA50 (trend change)
            if close[i] > r1_4h_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals