#!/usr/bin/env python3
"""
4h_1D_Camarilla_R3_S3_Breakout_RSI_Filter
Hypothesis: Breakout above/below daily Camarilla R3/S3 with RSI momentum filter (2-period RSI > 80 for long, < 20 for short) and volume confirmation.
Works in both bull and bear markets by capturing strong momentum moves with confirmation. Targets 25-35 trades/year to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: Range = (H-L), R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate 2-period RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 2:
            if i == 0:
                avg_gain[i] = gain[0]
                avg_loss[i] = loss[0]
            else:
                avg_gain[i] = (avg_gain[i-1] + gain[i]) / 2
                avg_loss[i] = (avg_loss[i-1] + loss[i]) / 2
        else:
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 2)  # volume MA needs 20, RSI needs 2
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume (reduced from 2.0 to increase signal quality)
        volume_confirmation = vol_ratio > 1.5
        
        # RSI momentum filter: > 80 for long, < 20 for short
        rsi_long_filter = rsi[i] > 80
        rsi_short_filter = rsi[i] < 20
        
        if position == 0:
            # Long: break above daily R3 with volume and RSI momentum
            if volume_confirmation and rsi_long_filter and price > camarilla_r3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 with volume and RSI momentum
            elif volume_confirmation and rsi_short_filter and price < camarilla_s3_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily midpoint or RSI loses momentum
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price < daily_mid or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to daily midpoint or RSI loses momentum
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price > daily_mid or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_1D_Camarilla_R3_S3_Breakout_RSI_Filter"
timeframe = "4h"
leverage = 1.0