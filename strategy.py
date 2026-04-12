#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian(15) channels (tighter for fewer signals)
    high_15_1d = pd.Series(high_1d).rolling(window=15, min_periods=15).max().values
    low_15_1d = pd.Series(low_1d).rolling(window=15, min_periods=15).min().values
    
    # Calculate daily RSI(14) for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(volume_1d)
    vol_ma_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    high_15_1d_aligned = align_htf_to_ltf(prices, df_1d, high_15_1d)
    low_15_1d_aligned = align_htf_to_ltf(prices, df_1d, low_15_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_15_1d_aligned[i]) or np.isnan(low_15_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period daily volume MA
        vol_filter = volume[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_15_1d_aligned[i]
        short_breakout = close[i] < low_15_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold for entries
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Entry conditions: breakout + volume + RSI filter
        long_entry = long_breakout and vol_filter and rsi_not_overbought
        short_entry = short_breakout and vol_filter and rsi_not_oversold
        
        # Exit conditions: opposite breakout or RSI reversal
        long_exit = (close[i] < low_15_1d_aligned[i]) or (rsi_1d_aligned[i] > 75)
        short_exit = (close[i] > high_15_1d_aligned[i]) or (rsi_1d_aligned[i] < 25)
        
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

name = "4h_1d_donchian_rsi_filter_v1"
timeframe = "4h"
leverage = 1.0