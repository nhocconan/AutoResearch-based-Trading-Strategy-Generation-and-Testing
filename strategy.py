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
    
    # Get weekly data for 5-period EMA (trend filter) and 10-period high/low (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly 5-period EMA
    close_1w = df_1w['close'].values
    ema5_1w = pd.Series(close_1w).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema5_1w_aligned = align_htf_to_ltf(prices, df_1w, ema5_1w)
    
    # Calculate weekly 10-period high and low for Donchian channels
    high_10 = pd.Series(df_1w['high'].values).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(df_1w['low'].values).rolling(window=10, min_periods=10).min().values
    high_10_aligned = align_htf_to_ltf(prices, df_1w, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1w, low_10)
    
    # Calculate 10-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = np.full(n, np.nan)
    for i in range(9, n):
        atr10[i] = np.nanmean(tr[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema5_1w_aligned[i]) or np.isnan(high_10_aligned[i]) or 
            np.isnan(low_10_aligned[i]) or np.isnan(atr10[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR10 > 0.8x ATR10 (always true, but keeps structure)
        vol_filter = True  # Simplified for 12h timeframe
        
        # Trend filter: price above/below weekly 5 EMA
        price_above_ema5 = close[i] > ema5_1w_aligned[i]
        price_below_ema5 = close[i] < ema5_1w_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend
        long_breakout = close[i] > high_10_aligned[i]  # break above weekly 10-period high
        short_breakout = close[i] < low_10_aligned[i]  # break below weekly 10-period low
        
        long_entry = long_breakout and price_above_ema5 and vol_filter
        short_entry = short_breakout and price_below_ema5 and vol_filter
        
        # Exit conditions: reversal signal
        long_exit = close[i] < ema5_1w_aligned[i]
        short_exit = close[i] > ema5_1w_aligned[i]
        
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

name = "12h_1w_donchian_ema5_breakout_v1"
timeframe = "12h"
leverage = 1.0