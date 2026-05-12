#!/usr/bin/env python3
name = "1d_Vortex_T1_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly volume filter: volume > 1.5x 20-period average
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Daily price data for Vortex Indicator (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.inf], tr1])
    vm = np.abs(high - np.roll(low, 1))
    vi = np.abs(low - np.roll(high, 1))
    
    # VI+ and VI- calculation
    vi_plus = pd.Series(vm).rolling(window=14, min_periods=14).sum().values / pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vi).rolling(window=14, min_periods=14).sum().values / pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # need enough data for Vortex
    
    for i in range(start_idx, n):
        # Skip if weekly trend or volume data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: daily volume > weekly average volume
        if volume[i] <= vol_ma_20_1w_aligned[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) and price above weekly EMA
            if vi_plus[i] > vi_minus[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) and price below weekly EMA
            elif vi_minus[i] > vi_plus[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when trend turns bearish or price crosses below EMA
            if vi_minus[i] > vi_plus[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when trend turns bullish or price crosses above EMA
            if vi_plus[i] > vi_minus[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals