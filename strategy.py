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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period SMA on 1w
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Volume confirmation on 6h: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: price breaks Donchian channels
        breakout_up = close[i] > high_20_aligned[i]
        breakout_down = close[i] < low_20_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_condition = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        
        # Weekly trend filter: only trade in direction of weekly trend
        above_weekly_sma = close[i] > sma_20_1w_aligned[i]
        below_weekly_sma = close[i] < sma_20_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and vol_condition and above_weekly_sma and vol_filter[i]
        short_entry = breakout_down and vol_condition and below_weekly_sma and vol_filter[i]
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (breakout_down or not vol_condition)
        exit_short = position == -1 and (breakout_up or not vol_condition)
        
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

name = "6h_1d_donchian_breakout_vol_filter"
timeframe = "6h"
leverage = 1.0