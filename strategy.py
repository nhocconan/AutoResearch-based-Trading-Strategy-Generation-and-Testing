#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1h price action for entry timing.
# Uses 1d pivot points (R1/S1) for structure, volume confirmation on 12h bars,
# and 1h trend filter (EMA20 > EMA50) to avoid counter-trend trades.
# Designed for fewer trades (<150/year) to minimize fee drag in both bull/bear markets.
name = "12h_1hEMA20_50_1dPivot_R1S1_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = 2 * pivot_1d - high_1d
    r1_1d = 2 * pivot_1d - low_1d
    
    # Align to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # 1h EMA trend filter (for entry direction)
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend = ema20 > ema50
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        s1 = s1_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        trend_up = uptrend[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Break above R1 with volume and uptrend
            if price > r1 and volume_confirmed and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and downtrend
            elif price < s1 and volume_confirmed and not trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or trend reverses
            if price < s1 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or trend reverses
            if price > r1 or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals