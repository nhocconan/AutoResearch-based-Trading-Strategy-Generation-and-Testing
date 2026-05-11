#!/usr/bin/env python3
name = "6h_EMA_RSI_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 60 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14) for momentum/divergence
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Weekly high/low for range context
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align all to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # 60-period RSI on 6c for overbought/oversold
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.clip(lower=0)
    loss_6h = -delta_6h.clip(upper=0)
    avg_gain_6h = gain_6h.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    avg_loss_6h = loss_6h.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rsi_6h = 100 - (100 / (1 + rs_6h))
    rsi_6h = rsi_6h.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(150, 60)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or
            np.isnan(rsi_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: oversold RSI on 6h + price above daily EMA50 + weekly support
            if (rsi_6h[i] < 30 and 
                close[i] > ema50_1d_aligned[i] and
                close[i] > low_1w_aligned[i] and  # above weekly low
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI on 6h + price below daily EMA50 + weekly resistance
            elif (rsi_6h[i] > 70 and 
                  close[i] < ema50_1d_aligned[i] and
                  close[i] < high_1w_aligned[i] and  # below weekly high
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or price breaks below daily EMA50
            if (rsi_6h[i] > 70 or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or price breaks above daily EMA50
            if (rsi_6h[i] < 30 or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals