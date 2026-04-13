#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Williams %R and RSI extremes + volume confirmation.
# Long: Williams %R < -80 (oversold) AND RSI < 30 (oversold) AND volume > 1.5x average volume.
# Short: Williams %R > -20 (overbought) AND RSI > 70 (overbought) AND volume > 1.5x average volume.
# Exit: Opposite condition (Williams %R crosses above -50 for long exit, below -50 for short exit).
# Uses 1d Williams %R for overbought/oversold extremes, RSI for confirmation, volume for conviction.
# Time filter: None (all hours) to capture opportunities in both bull and bear markets.
# Target: 20-50 total trades over 4 years (5-12.5/year) for 4h timeframe to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-14:i])
        lowest_low = np.min(low_1d[i-14:i])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close_1d[i-1]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral if no range
    
    # Calculate RSI (14-period)
    rsi = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Williams %R and RSI to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        wr = williams_r_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Williams %R < -80 AND RSI < 30 AND volume confirmation
            if (wr < -80 and rsi_val < 30 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R > -20 AND RSI > 70 AND volume confirmation
            elif (wr > -20 and rsi_val > 70 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum shifting)
            if wr > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum shifting)
            if wr < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Williams_RSI_Volume"
timeframe = "4h"
leverage = 1.0