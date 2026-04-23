#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses Donchian middle band (20-period SMA).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
The weekly EMA50 provides a robust trend filter that works in both bull and bear markets by avoiding counter-trend entries.
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
    
    # Load 1d data for Donchian levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # 20-period SMA of midpoint
    
    # Align Donchian levels to 1d timeframe (identity alignment since we're already on 1d)
    # But we still use align_htf_to_ltf to ensure proper handling of data gaps
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_band_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(mid_band_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band AND close > 1w EMA50 AND volume spike
            if (price > upper_band_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND close < 1w EMA50 AND volume spike
            elif (price < lower_band_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle band
                if price < mid_band_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle band
                if price > mid_band_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0