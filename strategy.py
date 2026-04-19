#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_BullBear_Power_With_Trend_v2"
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
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA on daily close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Daily High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Daily Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6-day EMA on 1d close for trend filter (longer-term trend)
    ema6_1d = pd.Series(close_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema6_aligned = align_htf_to_ltf(prices, df_1d, ema6_1d)
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30)
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema6_aligned[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND price above 6-day EMA (uptrend)
            if bull_power_aligned[i] > 0 and price > ema6_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND price below 6-day EMA (downtrend)
            elif bear_power_aligned[i] < 0 and price < ema6_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power <= 0 (bulls lose control) OR price below 6-day EMA
            if bull_power_aligned[i] <= 0 or price < ema6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power >= 0 (bears lose control) OR price above 6-day EMA
            if bear_power_aligned[i] >= 0 or price > ema6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals