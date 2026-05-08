#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# Long when price breaks above Donchian upper band + ATR rising + volume spike
# Short when price breaks below Donchian lower band + ATR rising + volume spike
# Exit when price crosses Donchian middle band or ATR falls below threshold
# Uses tight entry conditions to limit trades to 20-30/year for low fee drag
# Works in both bull and bear by filtering breakouts with volatility expansion

name = "4h_Donchian_Breakout_ATR_Volume_Filter"
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
    
    # Get 1d data for ATR calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max_20 + low_min_20) / 2
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(donchian_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        # Find the most recent completed 1d bar
        idx_1d = len(df_1d) - 1
        while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
            idx_1d -= 1
        vol_filter = False
        if idx_1d >= 0:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.5 * vol_ma_20_aligned[i]
        
        # ATR filter: current ATR > 1.2x 20-period ATR average
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean()
        atr_filter = atr_14_aligned[i] > 1.2 * atr_ma_20.iloc[i] if not np.isnan(atr_ma_20.iloc[i]) else False
        
        if position == 0:
            # Look for breakout with volatility expansion and volume confirmation
            # Long: price breaks above Donchian upper + volatility expansion + volume spike
            if close[i] > high_max_20[i] and atr_filter:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian lower + volatility expansion + volume spike
            elif close[i] < low_min_20[i] and atr_filter:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle or volatility contraction
            if close[i] < donchian_middle[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle or volatility contraction
            if close[i] > donchian_middle[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals