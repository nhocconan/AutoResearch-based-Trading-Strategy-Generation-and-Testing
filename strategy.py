#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1H data for RSI calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI on 1H
    delta_1h = pd.Series(df_1h['close']).diff()
    gain_1h = delta_1h.where(delta_1h > 0, 0)
    loss_1h = -delta_1h.where(delta_1h < 0, 0)
    avg_gain_1h = gain_1h.rolling(window=14, min_periods=14).mean()
    avg_loss_1h = loss_1h.rolling(window=14, min_periods=14).mean()
    rs_1h = avg_gain_1h / avg_loss_1h
    rsi_1h = 100 - (100 / (1 + rs_1h))
    rsi_1h_values = rsi_1h.values
    
    # Calculate RSI on 1D
    delta_1d = pd.Series(df_1d['close']).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.rolling(window=14, min_periods=14).mean()
    avg_loss_1d = loss_1d.rolling(window=14, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.values
    
    # Calculate RSI on 1W
    delta_1w = pd.Series(df_1w['close']).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.rolling(window=14, min_periods=14).mean()
    avg_loss_1w = loss_1w.rolling(window=14, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_values = rsi_1w.values
    
    # Align all data to 6H timeframe
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_values)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_1h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_1h_val = rsi_1h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_1w_val = rsi_1w_aligned[i]
        
        # Entry conditions: Multi-timeframe RSI alignment for mean reversion
        # Long when all timeframes show oversold conditions (RSI < 30)
        # Short when all timeframes show overbought conditions (RSI > 70)
        oversold_all = (rsi_1h_val < 30) and (rsi_1d_val < 30) and (rsi_1w_val < 30)
        overbought_all = (rsi_1h_val > 70) and (rsi_1d_val > 70) and (rsi_1w_val > 70)
        
        # Exit conditions: RSI returns to neutral territory (40-60)
        rsi_normal = (rsi_1h_val >= 40) and (rsi_1h_val <= 60)
        
        if position == 0:
            if oversold_all:
                position = 1
                signals[i] = position_size
            elif overbought_all:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when RSI returns to normal
            if rsi_normal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when RSI returns to normal
            if rsi_normal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1h1d1w_RSI_Multi_Timeframe_Mean_Reversion_v1"
timeframe = "6h"
leverage = 1.0