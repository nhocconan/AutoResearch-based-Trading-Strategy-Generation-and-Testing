#!/usr/bin/env python3
"""
1d_1w_RSI_Momentum_Breakout
Hypothesis: On daily timeframe, use weekly RSI (14) as trend filter and daily RSI (14) for momentum entries.
Long when weekly RSI > 50 (bullish regime) and daily RSI crosses above 60 with volume > 1.5x 20-day average.
Short when weekly RSI < 50 (bearish regime) and daily RSI crosses below 40 with volume > 1.5x 20-day average.
Exit when daily RSI crosses back to 50 or volume drops below average.
Uses weekly trend filter to avoid counter-trend trades in both bull and bear markets.
Target: 15-25 trades/year to minimize fee drag.
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
    
    # Load weekly data for trend filter (RSI)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI (14)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.full_like(close_1w, np.nan)
    avg_loss_1w = np.full_like(close_1w, np.nan)
    
    # Wilder's smoothing
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain_1w[i] = np.mean(gain_1w[1:15])
            avg_loss_1w[i] = np.mean(loss_1w[1:15])
        else:
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rs_1w = np.where(avg_loss_1w != 0, avg_gain_1w / avg_loss_1w, 0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Load daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI (14)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.full_like(close_1d, np.nan)
    avg_loss_1d = np.full_like(close_1d, np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain_1d[i] = np.mean(gain_1d[1:15])
            avg_loss_1d[i] = np.mean(loss_1d[1:15])
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate 20-day average volume
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # RSI needs warmup, volume MA needs 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        volume_ratio = volume_1d_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: regime filter + momentum + volume confirmation
            # Bullish: weekly RSI > 50 (bullish regime) AND daily RSI crosses above 60 AND volume > 1.5x average
            if (rsi_1w_aligned[i] > 50 and 
                rsi_1d_aligned[i] > 60 and 
                rsi_1d_aligned[i-1] <= 60 and  # Cross above 60
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Bearish: weekly RSI < 50 (bearish regime) AND daily RSI crosses below 40 AND volume > 1.5x average
            elif (rsi_1w_aligned[i] < 50 and 
                  rsi_1d_aligned[i] < 40 and 
                  rsi_1d_aligned[i-1] >= 40 and  # Cross below 40
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: daily RSI crosses below 50 OR volume drops below average
            if (rsi_1d_aligned[i] < 50 and rsi_1d_aligned[i-1] >= 50) or \
               volume_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: daily RSI crosses above 50 OR volume drops below average
            if (rsi_1d_aligned[i] > 50 and rsi_1d_aligned[i-1] <= 50) or \
               volume_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_RSI_Momentum_Breakout"
timeframe = "1d"
leverage = 1.0