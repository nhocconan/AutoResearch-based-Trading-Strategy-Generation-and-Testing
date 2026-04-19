#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_RSI4060_Confluence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(14) - higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Weekly RSI(14) - higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    delta_w = np.diff(close_1w, prepend=close_1w[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False).mean().values
    rs_w = avg_gain_w / (avg_loss_w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_w))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: RSI between 40-60 on both timeframes (neutral momentum) + volume
            if (40 <= rsi_1d_aligned[i] <= 60) and (40 <= rsi_1w_aligned[i] <= 60) and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI outside 40-60 on both timeframes (extreme momentum) + volume
            elif ((rsi_1d_aligned[i] < 40 or rsi_1d_aligned[i] > 60) and 
                  (rsi_1w_aligned[i] < 40 or rsi_1w_aligned[i] > 60) and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI moves above 60 or below 40 on either timeframe
            if (rsi_1d_aligned[i] > 60 or rsi_1d_aligned[i] < 40 or 
                rsi_1w_aligned[i] > 60 or rsi_1w_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI moves into 40-60 range on either timeframe
            if (40 <= rsi_1d_aligned[i] <= 60) or (40 <= rsi_1w_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals