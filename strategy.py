#!/usr/bin/env python3
name = "6h_1d_RSI_Volume_Momentum"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 60-period RSI for momentum (momentum oscillator)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    avg_loss = loss.ewm(alpha=1/60, adjust=False, min_periods=60).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily RSI for trend filter (40-60 range indicates trend strength)
    df_1d_close = pd.Series(df_1d['close'])
    delta_1d = df_1d_close.diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.values
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 60)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + daily RSI > 40 (uptrend bias) + volume surge
            if (rsi_values[i] < 30 and 
                rsi_1d_aligned[i] > 40 and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + daily RSI < 60 (downtrend bias) + volume surge
            elif (rsi_values[i] > 70 and 
                  rsi_1d_aligned[i] < 60 and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or daily trend weakens
            if (rsi_values[i] > 70 or rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or daily trend weakens
            if (rsi_values[i] < 30 or rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals