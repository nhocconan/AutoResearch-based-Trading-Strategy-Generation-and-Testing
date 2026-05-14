#!/usr/bin/env python3
"""
1h_4h_1d_rsi_volume_v1
Hypothesis: Use RSI(14) on 4h for trend direction and RSI(14) on 1d for momentum filter, with 1h for entry timing and volume confirmation.
Long when 4h RSI > 50 and 1d RSI > 50 and price crosses above 1h VWAP with volume > 1.5x average.
Short when 4h RSI < 50 and 1d RSI < 50 and price crosses below 1h VWAP with volume > 1.5x average.
Designed to work in both bull and bear markets by using multi-timeframe RSI alignment.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) by requiring multi-timeframe alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Get 1d data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on 4h
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h[:13] = np.nan  # Not enough data for first 14 periods
    
    # Calculate RSI on 1d
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d[:13] = np.nan  # Not enough data for first 14 periods
    
    # Align RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate VWAP on 1h
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=14, min_periods=14).sum().values
    vwap_den = pd.Series(volume).rolling(window=14, min_periods=14).sum().values
    vwap = vwap_num / (vwap_den + 1e-10)
    vwap[:13] = np.nan  # Not enough data for first 14 periods
    
    # Volume confirmation: volume > 1.5x average of last 14 periods
    vol_ma = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Determine multi-timeframe alignment
        bullish_alignment = rsi_4h_aligned[i] > 50 and rsi_1d_aligned[i] > 50
        bearish_alignment = rsi_4h_aligned[i] < 50 and rsi_1d_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP or alignment breaks
            if close[i] < vwap[i] or not bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP or alignment breaks
            if close[i] > vwap[i] or not bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish alignment and price crosses above VWAP with volume
            if bullish_alignment and close[i] > vwap[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: bearish alignment and price crosses below VWAP with volume
            elif bearish_alignment and close[i] < vwap[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.20
    
    return signals