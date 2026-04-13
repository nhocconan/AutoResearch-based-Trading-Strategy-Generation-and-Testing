#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume
Hypothesis: Uses 1d Camarilla levels (H4/L4) on 4h timeframe with volume confirmation and ADX filter.
Enters long when 4h close > H4 and volume > 1.5x 20-period average and ADX > 25.
Enters short when 4h close < L4 and volume > 1.5x 20-period average and ADX > 25.
Exits when price returns to prior 1d close.
Designed for 4h timeframe to target 19-50 trades/year (75-200 total over 4 years).
Works in both bull and bear markets by requiring volume expansion and trend strength on breakouts.
"""

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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous 1d bar
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.125 * hl_range
    L4 = close_1d - 1.125 * hl_range
    
    # Calculate 20-period volume average on 1d
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 4h for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(high[1:] - close[:-1], np.maximum(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Prepend NaN for alignment
    plus_di = np.concatenate([np.full(1, np.nan), plus_di])
    minus_di = np.concatenate([np.full(1, np.nan), minus_di])
    adx = np.concatenate([np.full(14, np.nan), adx[14:]])
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 1d volume MA
        volume_expansion = volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        # Trend strength: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions: price CLOSES beyond H4/L4 with volume expansion and strong trend
        long_entry = (close[i] > H4_aligned[i]) and volume_expansion and strong_trend
        short_entry = (close[i] < L4_aligned[i]) and volume_expansion and strong_trend
        
        # Exit conditions: return to previous 1d close
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and close[i] <= prev_close_aligned[i]
        exit_short = position == -1 and close[i] >= prev_close_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0