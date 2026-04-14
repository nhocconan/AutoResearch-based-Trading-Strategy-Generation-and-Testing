#!/usr/bin/env python3
"""
6h_1d_RSI_MultiTF_Trend_v1
Hypothesis: On 6h timeframe, use 1d RSI as trend filter (RSI>50 for long bias, RSI<50 for short bias) 
and 6s RSI pullback entries with volume confirmation. RSI on higher timeframe is less noisy and 
more reliable for trend direction, while lower timeframe provides better entry timing. 
Volume confirmation ensures institutional participation. Designed to work in both bull and bear 
markets by following the higher timeframe momentum.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI on 1d data (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Load 6s data for RSI entry signal and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate RSI on 6h data (14-period)
    delta_6h = np.diff(close_6h, prepend=close_6h[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(close_6h)
    avg_loss_6h = np.zeros_like(close_6h)
    avg_gain_6h[13] = np.mean(gain_6h[1:14])
    avg_loss_6h[13] = np.mean(loss_6h[1:14])
    
    for i in range(14, len(close_6h)):
        avg_gain_6h[i] = (avg_gain_6h[i-1] * 13 + gain_6h[i]) / 14
        avg_loss_6h[i] = (avg_loss_6h[i-1] * 13 + loss_6h[i]) / 14
    
    rs_6h = np.divide(avg_gain_6h, avg_loss_6h, out=np.full_like(avg_gain_6h, np.nan), where=avg_loss_6h!=0)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # Calculate 20-period average volume on 6h data
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(19, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    rsi_6h_aligned = align_htf_to_ltf(prices, df_6h, rsi_6h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # RSI needs ~2*period for stability, volume MA needs 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_6h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_ratio = volume_6h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: follow 1d RSI trend with 6h RSI pullback + volume
            # Long: 1d RSI > 50 (bullish trend) AND 6h RSI < 40 (pullback) AND volume > 1.5x average
            if (rsi_1d_aligned[i] > 50 and 
                rsi_6h_aligned[i] < 40 and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: 1d RSI < 50 (bearish trend) AND 6h RSI > 60 (pullback) AND volume > 1.5x average
            elif (rsi_1d_aligned[i] < 50 and 
                  rsi_6h_aligned[i] > 60 and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 1d RSI < 50 (trend change) OR 6h RSI > 70 (overbought)
            if rsi_1d_aligned[i] < 50 or rsi_6h_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 1d RSI > 50 (trend change) OR 6h RSI < 30 (oversold)
            if rsi_1d_aligned[i] > 50 or rsi_6h_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_RSI_MultiTF_Trend_v1"
timeframe = "6h"
leverage = 1.0