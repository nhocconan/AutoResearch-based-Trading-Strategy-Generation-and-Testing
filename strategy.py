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
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 10-period EMA (trend filter)
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 6-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr6 = np.full(n, np.nan)
    for i in range(5, n):
        atr6[i] = np.nanmean(tr[i-5:i+1])
    
    # Calculate 6-period ATR EMA for volatility regime
    atr_ema6 = np.full(n, np.nan)
    atr_series = pd.Series(atr6)
    atr_ema6_values = atr_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    atr_ema6[:] = atr_ema6_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(atr6[i]) or 
            np.isnan(atr_ema6[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR6 > 1.2x 6-period ATR EMA (elevated volatility)
        vol_filter = atr6[i] > atr_ema6[i] * 1.2
        
        # Trend filter: price above/below weekly 10 EMA
        price_above_ema10 = close[i] > ema10_1w_aligned[i]
        price_below_ema10 = close[i] < ema10_1w_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high_20_aligned[i]  # break above daily 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below daily 20-period low
        
        long_entry = long_breakout and price_above_ema10 and vol_filter
        short_entry = short_breakout and price_below_ema10 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema10_1w_aligned[i]) or (atr6[i] < atr_ema6[i] * 0.8)
        short_exit = (close[i] > ema10_1w_aligned[i]) or (atr6[i] < atr_ema6[i] * 0.8)
        
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

name = "6h_1w_donchian_ema10_vol_filter_v1"
timeframe = "6h"
leverage = 1.0