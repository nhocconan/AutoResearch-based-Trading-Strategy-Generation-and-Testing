#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w pivot levels for directional bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 12h Camarilla pivot levels (R1, S1)
    high_12h = high
    low_12h = low
    close_12h = close
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    r1_12h = pivot_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = pivot_12h - (high_12h - low_12h) * 1.1 / 12
    
    # 12h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.5x average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Entry conditions: Camarilla R1/S1 breakout with trend and volume
        if position == 0:
            # Long: price breaks above R1 with uptrend bias and volume
            if price > r1_12h[i] and price > ema200_1d_aligned[i] and price > pivot_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downtrend bias and volume
            elif price < s1_12h[i] and price < ema200_1d_aligned[i] and price < pivot_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot or stops below entry
            if price <= pivot_12h[i] or price < close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot or stops above entry
            if price >= pivot_12h[i] or price > close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals