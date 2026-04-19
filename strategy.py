#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_r1_s1_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot levels from previous weekly bar
    prev_close_weekly = np.roll(close_weekly, 1)
    prev_close_weekly[0] = np.nan
    prev_high_weekly = np.roll(high_weekly, 1)
    prev_high_weekly[0] = np.nan
    prev_low_weekly = np.roll(low_weekly, 1)
    prev_low_weekly[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_weekly = (prev_high_weekly + prev_low_weekly + prev_close_weekly) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_weekly = prev_close_weekly + (prev_high_weekly - prev_low_weekly) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_weekly = prev_close_weekly - (prev_high_weekly - prev_low_weekly) * 1.1 / 12.0
    
    # Align to daily timeframe
    pivot_weekly_1d = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_weekly_1d = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_1d = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_weekly_1d[i]) or np.isnan(r1_weekly_1d[i]) or np.isnan(s1_weekly_1d[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume spike
            if price > r1_weekly_1d[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume spike
            elif price < s1_weekly_1d[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly S1 (reversal signal)
            if price < s1_weekly_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly R1 (reversal signal)
            if price > r1_weekly_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals