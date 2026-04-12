#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d/1w trend filter and volume confirmation
    # Uses 1d EMA50 for intermediate trend and 1w EMA200 for long-term regime
    # Breakouts only taken in direction of both 1d and 1w trends to avoid counter-trend whipsaws
    # Volume confirmation (>1.5x 20-period average) filters low-conviction moves
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for intermediate trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data for long-term regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w EMA200 for regime filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian channels (20-period) on 6h data
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Align all indicators to LTF (6h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Trend filters
        bullish_1d = close[i] > ema50_1d_aligned[i]
        bearish_1d = close[i] < ema50_1d_aligned[i]
        bullish_1w = close[i] > ema200_1w_aligned[i]
        bearish_1w = close[i] < ema200_1w_aligned[i]
        
        # Volume confirmation (>1.5x 20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma)
        else:
            volume_confirm = False
        
        # Entry logic: Breakout + trend alignment (both 1d and 1w) + volume confirmation
        long_entry = long_breakout and bullish_1d and bullish_1w and volume_confirm
        short_entry = short_breakout and bearish_1d and bearish_1w and volume_confirm
        
        # Exit logic: opposite Donchian breakout or loss of trend alignment
        long_exit = short_breakout or not bullish_1d or not bullish_1w
        short_exit = long_breakout or not bearish_1d or not bearish_1w
        
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

name = "6h_1d_1w_donchian_breakout_ema50_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0