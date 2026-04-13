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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate previous daily bar's Camarilla levels (H3, L3)
    camarilla_h3_1d = np.full(len(high_1d), np.nan)
    camarilla_l3_1d = np.full(len(low_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        camarilla_h3_1d[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3_1d[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Get weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily volume for confirmation (>2.0x 20-period average)
    vol_ma_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    volume_spike_1d = volume > (2.0 * vol_ma_1d)
    
    # Align all indicators to LTF (1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h3_1d_aligned[i]
        short_breakout = close[i] < l3_1d_aligned[i]
        
        # Weekly trend filter (EMA50)
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_1d[i]
        short_entry = short_breakout and bearish_trend and volume_spike_1d[i]
        
        # Exit logic: price retests H3/L3 or trend reversal
        long_exit = (close[i] <= h3_1d_aligned[i] * 1.001) or not bullish_trend  # Retest H3 or trend change
        short_exit = (close[i] >= l3_1d_aligned[i] * 0.999) or not bearish_trend  # Retest L3 or trend change
        
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

name = "1d_1w_camarilla_h3l3_ema50_volume_v1"
timeframe = "1d"
leverage = 1.0