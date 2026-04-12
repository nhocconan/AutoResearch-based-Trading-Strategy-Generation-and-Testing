#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volatility filter and volume confirmation
    # Long: price breaks above Donchian(20) high + ATR(14) < 0.5 * ATR(50) (low vol) + volume > 1.3x MA
    # Short: price breaks below Donchian(20) low + same vol/vol filters
    # Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) and ATR(50) on 1d
    tr1 = np.zeros(len(df_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr1).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Donchian channels (20-period) on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) < 0.5 * ATR(50) (low volatility environment)
        low_volatility = atr_14_1d_aligned[i] < 0.5 * atr_50_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        # Entry conditions with volume and volatility confirmation
        long_entry = breakout_up and low_volatility and (vol_ratio[i] > 1.3)
        short_entry = breakout_down and low_volatility and (vol_ratio[i] > 1.3)
        
        # Exit conditions: opposite Donchian breakout
        long_exit = close[i] < lowest_20[i]
        short_exit = close[i] > highest_20[i]
        
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

name = "12h_1d_donchian_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0