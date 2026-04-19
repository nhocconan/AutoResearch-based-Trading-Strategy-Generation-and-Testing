#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels from previous week
    prev_close = np.roll(close_1w, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1w, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1w, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Time filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma_50[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_50[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 with volume
            if price > r1_6h[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume
            elif price < s1_6h[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal)
            if price < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal)
            if price > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals