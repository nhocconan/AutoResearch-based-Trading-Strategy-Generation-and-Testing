#!/usr/bin/env python3
"""
4h Camarilla Pivot R1S1 Breakout with Volume Confirmation and 1D EMA Trend Filter
Long: Price breaks above R1 + volume > 1.5x 4h volume MA + price > 1D EMA50
Short: Price breaks below S1 + volume > 1.5x 4h volume MA + price < 1D EMA50
Exit: Opposite break of S1/R1 or 1D EMA50 crossover
Uses proven Camarilla pivot structure with volume and trend filters to reduce false breakouts
Target: 25-35 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price for previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Range
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pp + (range_hl * 1.1 / 12)
    s1 = pp - (range_hl * 1.1 / 12)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1D EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume moving average (20-period for confirmation)
    df_4h = get_htf_data(prices, '4h')
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_aligned[i]
        
        if position == 0:
            # Long: break above R1 + volume + 1D trend
            if price > r1_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 + volume + 1D trend
            elif price < s1_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below S1 or trend reversal
            if price < s1_aligned[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 or trend reversal
            if price > r1_aligned[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_1DEMA50"
timeframe = "4h"
leverage = 1.0