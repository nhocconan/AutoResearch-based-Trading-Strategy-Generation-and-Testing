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
    
    # Get daily data for 20-period EMA (trend filter) and 20-period high/low (Donchian channels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily 20-period EMA
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate daily 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 10-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = np.full(n, np.nan)
    for i in range(9, n):
        atr10[i] = np.nanmean(tr[i-9:i+1])
    
    # Calculate 20-period ATR EMA for volatility regime
    atr_ema20 = np.full(n, np.nan)
    atr_series = pd.Series(atr10)
    atr_ema20_values = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_ema20[:] = atr_ema20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(atr10[i]) or 
            np.isnan(atr_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR10 > 1.0x 20-period ATR EMA (elevated volatility)
        vol_filter = atr10[i] > atr_ema20[i] * 1.0
        
        # Trend filter: price above/below daily 20 EMA
        price_above_ema20 = close[i] > ema20_1d_aligned[i]
        price_below_ema20 = close[i] < ema20_1d_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high_20_aligned[i]  # break above daily 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below daily 20-period low
        
        long_entry = long_breakout and price_above_ema20 and vol_filter
        short_entry = short_breakout and price_below_ema20 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema20_1d_aligned[i]) or (atr10[i] < atr_ema20[i] * 0.8)
        short_exit = (close[i] > ema20_1d_aligned[i]) or (atr10[i] < atr_ema20[i] * 0.8)
        
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

name = "4h_1d_donchian_ema20_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0