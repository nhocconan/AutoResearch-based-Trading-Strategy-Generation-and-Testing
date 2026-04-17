#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d trend filter and 1w momentum filter.
Trade 6h breakouts of Donchian channels with 1d EMA50 trend filter and 1w RSI momentum filter.
Use 6h for structure, 1d for trend direction, 1w for momentum confirmation.
Designed to work in both bull and bear markets by requiring trend alignment and momentum confirmation,
reducing false breakouts and whipsaws.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 6h data for structure (Donchian channels)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w RSI(14) for momentum filter
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Align all timeframes to 6h
    high_max_20_aligned = align_htf_to_ltf(prices, df_6h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_6h, low_min_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_values)
    
    # Volume filter: current volume > 1.5x 24-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_aligned[i]) or
            np.isnan(vol_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian high with volume, above 1d EMA50, and 1w RSI > 50
            if (close[i] > high_max_20_aligned[i] and volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i] and rsi_14_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low with volume, below 1d EMA50, and 1w RSI < 50
            elif (close[i] < low_min_20_aligned[i] and volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i] and rsi_14_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 6h Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 6h Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_1wRSI50"
timeframe = "6h"
leverage = 1.0