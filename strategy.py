#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA200 trend filter and volume confirmation (>2.0x average)
    # Uses 1d EMA200 for stronger trend filter to reduce whipsaws and overtrading
    # Volume spike >2.0x 20-period average confirms institutional participation
    # Target: 15-25 trades/year (60-100 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate previous 4h bar's Camarilla levels (H3, L3)
    camarilla_h3 = np.full(len(high_4h), np.nan)
    camarilla_l3 = np.full(len(low_4h), np.nan)
    
    for i in range(1, len(high_4h)):
        ph = high_4h[i-1]
        pl = low_4h[i-1]
        pc = close_4h[i-1]
        rang = ph - pl
        
        camarilla_h3[i] = pc + rang * 1.1 / 4  # H3 level
        camarilla_l3[i] = pc - rang * 1.1 / 4  # L3 level
    
    # Get 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 4h volume for confirmation (>2.0x 20-period average)
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    volume_spike_4h = volume > (2.0 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h3_4h_aligned[i]
        short_breakout = close[i] < l3_4h_aligned[i]
        
        # 1d trend filter (EMA200)
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h[i]
        
        # Exit logic: price retests H3/L3 or trend reversal
        long_exit = (close[i] <= h3_4h_aligned[i] * 1.001) or not bullish_trend
        short_exit = (close[i] >= l3_4h_aligned[i] * 0.999) or not bearish_trend
        
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

name = "4h_1d_camarilla_h3l3_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0