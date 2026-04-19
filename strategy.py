#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_R1S1_Breakout_Volume_Filter_v2"
timeframe = "4h"
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
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    # Camarilla levels: R4, R3, R2, R1, S1, S2, S3, S4
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    cam_multiplier = 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * cam_multiplier
    s1 = prev_close - (prev_high - prev_low) * cam_multiplier
    r4 = prev_close + (prev_high - prev_low) * (1.1 / 2)
    s4 = prev_close - (prev_high - prev_low) * (1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = np.mean(volume[max(0, i-20):i+1])  # 20-period average volume
        vol_filter = vol > 1.5 * avg_vol  # Volume spike filter
        
        # Entry conditions
        # Long: price breaks above R1 with volume and trend bias
        # Short: price breaks below S1 with volume and trend bias
        
        long_breakout = price > r1[i-1] and vol_filter and price > ema200_1d_aligned[i] and price > pivot_1w_aligned[i]
        short_breakout = price < s1[i-1] and vol_filter and price < ema200_1d_aligned[i] and price < pivot_1w_aligned[i]
        
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches R4 or stops below S1
            if price >= r4[i] or price < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches S4 or stops above R1
            if price <= s4[i] or price > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals