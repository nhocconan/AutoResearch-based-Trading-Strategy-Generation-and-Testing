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
    
    # Get daily data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility and stop
    tr1_d = np.abs(high_1d - low_1d)
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = tr2_d[0] = tr3_d[0] = np.nan
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_d[i-14:i+1])
    
    # Calculate daily high/low for Donchian channel (20-period)
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(volume_1d)
    vol_ma_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(high_20d_aligned[i]) or 
            np.isnan(low_20d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5 * 20-period daily MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20d_aligned[i]
        short_breakout = close[i] < low_20d_aligned[i]
        
        # Entry conditions: breakout + volume filter
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # ATR-based stop loss
        if position == 1:
            # For long, stop if price drops below entry - 2*ATR (approximated)
            # We use close-based exit: if close drops significantly
            if close[i] < close[i-1] - 2.0 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # For short, stop if price rises above entry + 2*ATR
            if close[i] > close[i-1] + 2.0 * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            # No position, check for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_vol_filter_v2"
timeframe = "12h"
leverage = 1.0