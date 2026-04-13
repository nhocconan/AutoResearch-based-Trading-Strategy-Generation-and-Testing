#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout + 1d ATR volatility filter + volume confirmation
    # Long: price > Donchian High(20) AND ATR(14) > ATR(50) AND volume > 1.5x avg
    # Short: price < Donchian Low(20) AND ATR(14) > ATR(50) AND volume > 1.5x avg
    # Exit: price crosses Donchian midpoint OR ATR(14) < ATR(50) (vol collapse)
    # Using 4h timeframe for proper trade frequency, Donchian for structure,
    # 1d ATR ratio for volatility regime filter (avoid low-vol chop), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr14[i] = np.nanmean(tr[1:15])  # skip index 0 (NaN)
        else:
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # ATR(50)
    atr50 = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        if i == 50:
            atr50[i] = np.nanmean(tr[1:51])
        else:
            atr50[i] = (atr50[i-1] * 49 + tr[i]) / 50
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr50 > 0, atr14 / atr50, 0)
    
    # Align ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donch_high[i] = np.max(high[i-lookback:i])
        donch_low[i] = np.min(low[i-lookback:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: ATR ratio > 1.0 = expanding volatility (good for breakouts)
        vol_expanding = atr_ratio_aligned[i] > 1.0
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + vol expanding + volume confirmation
        long_entry = (close[i] > donch_high[i]) and vol_expanding and vol_confirm
        short_entry = (close[i] < donch_low[i]) and vol_expanding and vol_confirm
        
        # Exit logic: price crosses midpoint OR volatility collapses
        long_exit = (close[i] < donch_mid[i]) or (atr_ratio_aligned[i] < 0.8)
        short_exit = (close[i] > donch_mid[i]) or (atr_ratio_aligned[i] < 0.8)
        
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

name = "4h_1d_donchian_atr_volume_v2"
timeframe = "4h"
leverage = 1.0