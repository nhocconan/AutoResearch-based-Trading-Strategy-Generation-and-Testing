#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_PowerTrend_v1"
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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Elder Ray Power (Bull/Bear Power) on 1d
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 13-period EMA of close for trend filter
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly trend filter: price above/below weekly EMA26
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or \
           np.isnan(ema13_6h[i]) or np.isnan(ema26_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filters
        price_above_ema13 = price > ema13_6h[i]
        price_below_ema13 = price < ema13_6h[i]
        weekly_uptrend = price > ema26_1w_aligned[i]
        weekly_downtrend = price < ema26_1w_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive AND price above EMA13 AND weekly uptrend AND volume
            if bull_power > 0 and price_above_ema13 and weekly_uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND price below EMA13 AND weekly downtrend AND volume
            elif bear_power < 0 and price_below_ema13 and weekly_downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative OR price breaks below EMA13
            if bull_power <= 0 or price <= ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive OR price breaks above EMA13
            if bear_power >= 0 or price >= ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals