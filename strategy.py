#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d ATR for volatility filter
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.absolute(high_1d - np.roll(close_1d, 1)), 
                       np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Camarilla pivot levels (R1 and S1)
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = close_1d + (1.1/12) * (prev_high_1d - prev_low_1d)
    s1_1d = close_1d - (1.1/12) * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 12h ATR for position sizing
    tr = np.maximum(high - low, 
                    np.absolute(high - np.roll(close, 1)), 
                    np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume moving average for volume confirmation
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or \
           np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        vol_ma = vol_ma_12h[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_ma
        
        # Trend bias: price above/below 1w EMA34
        long_bias = price > ema34_1w_aligned[i]
        short_bias = price < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and long bias
            if price > r1_1d_aligned[i] and volume_confirm and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and short bias
            elif price < s1_1d_aligned[i] and volume_confirm and short_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below pivot or ATR-based stop
            if price < pivot_1d_aligned[i] or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above pivot or ATR-based stop
            if price > pivot_1d_aligned[i] or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals