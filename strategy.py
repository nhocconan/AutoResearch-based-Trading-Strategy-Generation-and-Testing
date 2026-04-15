#!/usr/bin/env python3
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
    
    # Load weekly data once
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 50:
        return np.zeros(n)
    
    # Weekly ATR(14)
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    close_week = df_week['close'].values
    tr1 = high_week - low_week
    tr2 = np.abs(high_week - np.roll(close_week, 1))
    tr3 = np.abs(low_week - np.roll(close_week, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr_week = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly Donchian(20)
    high_20_week = pd.Series(high_week).rolling(window=20, min_periods=20).max().values
    low_20_week = pd.Series(low_week).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(50)
    ema_50_week = pd.Series(close_week).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned weekly indicators
        atr_week_aligned = align_htf_to_ltf(prices, df_week, atr_week)[i]
        high_20_week_aligned = align_htf_to_ltf(prices, df_week, high_20_week)[i]
        low_20_week_aligned = align_htf_to_ltf(prices, df_week, low_20_week)[i]
        ema_50_week_aligned = align_htf_to_ltf(prices, df_week, ema_50_week)[i]
        
        # Check for NaN values
        if (np.isnan(atr_week_aligned) or np.isnan(high_20_week_aligned) or 
            np.isnan(low_20_week_aligned) or np.isnan(ema_50_week_aligned)):
            continue
        
        # Volume filter (volume > 1.5x 20-period average)
        vol_ma_20 = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else volume[i]
        volume_filter = volume[i] > 1.5 * vol_ma_20
        
        if position == 0:  # No position - look for entries
            if volume_filter:
                # Long: Break above weekly Donchian high + above weekly EMA50
                if high[i] > high_20_week_aligned and close[i] > ema_50_week_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: Break below weekly Donchian low + below weekly EMA50
                elif low[i] < low_20_week_aligned and close[i] < ema_50_week_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below weekly Donchian low
            if low[i] < low_20_week_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above weekly Donchian high
            if high[i] > high_20_week_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyATR_Donchian20_EMA50_Volume"
timeframe = "12h"
leverage = 1.0