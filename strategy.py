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
    
    # Load 1d data once before loop (for 12h timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once before loop (for volatility filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility filter
    high_low_1w = df_1w['high'] - df_1w['low']
    high_close_1w = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    low_close_1w = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d RSI (14-period) for mean reversion
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]) or len(atr_1w) == 0:
            continue
        
        # Get 1d index for current 12h bar (12h = 0.5 * 1d)
        idx_1d = i // 2
        if idx_1d < 14:  # Need sufficient 1d data for RSI
            continue
            
        # Get 1w index for current 12h bar (12h = 1/14 * 1w approximately)
        idx_1w = i // 14
        if idx_1w < 14:  # Need sufficient 1w data for ATR
            continue
        
        # Previous values to avoid look-ahead
        rsi_prev = rsi_1d[idx_1d-1]
        atr_prev = atr_1w[idx_1w-1] if idx_1w-1 < len(atr_1w) else atr_1w[-1]
        
        if np.isnan(rsi_prev):
            continue
        
        # Create arrays for alignment (constant values for the period)
        rsi_arr = np.full(len(df_1d), rsi_prev)
        atr_arr = np.full(len(df_1w), atr_prev)
        
        # Align to 12h timeframe
        rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_arr)[i]
        atr_12h = align_htf_to_ltf(prices, df_1w, atr_arr)[i]
        
        if position == 0:
            # Long: RSI oversold (< 30) + volatility expansion (current ATR > 1.5 * average ATR) + volume confirmation
            if (rsi_12h < 30 and 
                atr_12h > 1.5 * atr_prev and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (> 70) + volatility expansion + volume confirmation
            elif (rsi_12h > 70 and 
                  atr_12h > 1.5 * atr_prev and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI returns to neutral (> 50) or volatility contraction
            if rsi_12h > 50 or atr_12h < 0.5 * atr_prev:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI returns to neutral (< 50) or volatility contraction
            if rsi_12h < 50 or atr_12h < 0.5 * atr_prev:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_RSI_1w_ATR_Volatility"
timeframe = "12h"
leverage = 1.0