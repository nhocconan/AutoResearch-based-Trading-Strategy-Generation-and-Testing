#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA 8 for trend direction
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1]))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 6h Donchian channels (20-period for structure)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    highest_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    highest_high_6h_aligned = align_htf_to_ltf(prices, df_6h, highest_high_6h)
    lowest_low_6h_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_6h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_8_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(highest_high_6h_aligned[i]) or 
            np.isnan(lowest_low_6h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA8
        price_above_ema = close[i] > ema_8_1w_aligned[i]
        price_below_ema = close[i] < ema_8_1w_aligned[i]
        
        # Volatility filter: only trade when ATR is above average (avoid choppy markets)
        vol_filter = atr_1w_aligned[i] > np.nanmedian(atr_1w_aligned[max(0, i-50):i+1])
        
        # Long conditions: price breaks above 6h Donchian high + above weekly EMA + volume + volatility
        long_breakout = (close[i] > highest_high_6h_aligned[i-1] and 
                        price_above_ema and 
                        volume_filter[i] and 
                        vol_filter)
        
        # Short conditions: price breaks below 6h Donchian low + below weekly EMA + volume + volatility
        short_breakout = (close[i] < lowest_low_6h_aligned[i-1] and 
                         price_below_ema and 
                         volume_filter[i] and 
                         vol_filter)
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite 6h Donchian breakout
        elif position == 1 and close[i] < lowest_low_6h_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_6h_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_1wEMA8_ATRVolFilter"
timeframe = "6h"
leverage = 1.0