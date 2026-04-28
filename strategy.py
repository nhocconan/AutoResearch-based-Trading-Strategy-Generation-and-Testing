#!/usr/bin/env python3
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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for long-term trend
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 6h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200 (long-term bias)
        trend_up = close[i] > ema_200_aligned[i]
        trend_down = close[i] < ema_200_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_ma[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_ma[i-1]  # Break below previous period's low
        
        # Volatility filter: only trade in normal volatility (avoid choppy markets)
        vol_normal = atr_14_aligned[i] < 1.5 * np.median(atr_14_aligned[max(0, i-50):i+1])
        
        # Volume confirmation: above average volume
        vol_confirm = volume[i] > vol_ma[i]
        
        # Entry conditions - require trend alignment + breakout + filters
        long_entry = trend_up and breakout_up and vol_normal and vol_confirm
        short_entry = trend_down and breakout_down and vol_normal and vol_confirm
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Breakout_EMA200_Trend_Filter"
timeframe = "6h"
leverage = 1.0