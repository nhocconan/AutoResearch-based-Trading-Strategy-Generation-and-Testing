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
    
    # Get daily data for 150-period EMA filter and 10-period ATR volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 150:
        return np.zeros(n)
    
    # Calculate daily 150-period EMA
    close_1d = df_1d['close'].values
    ema150_1d = pd.Series(close_1d).ewm(span=150, adjust=False, min_periods=150).mean().values
    ema150_1d_aligned = align_htf_to_ltf(prices, df_1d, ema150_1d)
    
    # Calculate daily 10-period ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = np.full(len(df_1d), np.nan)
    for i in range(9, len(df_1d)):
        atr10[i] = np.nanmean(tr[i-9:i+1])
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10)
    
    # Calculate 12-period ATR for current volatility
    tr1_12h = np.abs(high - low)
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = tr2_12h[0] = tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr12 = np.full(n, np.nan)
    for i in range(11, n):
        atr12[i] = np.nanmean(tr_12h[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema150_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i]) or 
            np.isnan(atr12[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 12h ATR > 1.3x daily ATR10 (high volatility regime)
        vol_filter = atr12[i] > atr10_1d_aligned[i] * 1.3
        
        # Trend filter: price above/below daily 150 EMA
        price_above_ema150 = close[i] > ema150_1d_aligned[i]
        price_below_ema150 = close[i] < ema150_1d_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volatility expansion
        long_breakout = close[i] > high[i-1]  # break above previous high
        short_breakout = close[i] < low[i-1]  # break below previous low
        
        long_entry = long_breakout and price_above_ema150 and vol_filter
        short_entry = short_breakout and price_below_ema150 and vol_filter
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema150_1d_aligned[i]) or (atr12[i] < atr10_1d_aligned[i] * 0.7)
        short_exit = (close[i] > ema150_1d_aligned[i]) or (atr12[i] < atr10_1d_aligned[i] * 0.7)
        
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

name = "12h_1d_ema150_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0