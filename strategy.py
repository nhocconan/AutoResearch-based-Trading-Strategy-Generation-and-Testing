#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d RSI-based mean reversion and volume confirmation.
# Long: RSI(14) < 30 on 1d + price > VWAP(20) on 12h + volume > 1.5x average volume.
# Short: RSI(14) > 70 on 1d + price < VWAP(20) on 12h + volume > 1.5x average volume.
# Uses 1d RSI for overbought/oversold conditions, 12h for execution with VWAP and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initial average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Wilder's smoothing
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.full(len(close_1d), np.nan)
    rsi = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100  # Avoid division by zero
    
    # VWAP(20) on 12h
    typical_price = (high + low + close) / 3
    vwap_num = np.full(n, np.nan)
    vwap_den = np.full(n, np.nan)
    
    for i in range(20, n):
        vwap_num[i] = np.sum(typical_price[i-20:i] * volume[i-20:i])
        vwap_den[i] = np.sum(volume[i-20:i])
    
    vwap = np.full(n, np.nan)
    for i in range(20, n):
        if vwap_den[i] != 0:
            vwap[i] = vwap_num[i] / vwap_den[i]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi_aligned[i]
        vwap_val = vwap[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price > VWAP + volume confirmation
            if (rsi_val < 30 and price > vwap_val and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + price < VWAP + volume confirmation
            elif (rsi_val > 70 and price < vwap_val and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (neutral) or price < VWAP
            if (rsi_val > 50 or price < vwap_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 (neutral) or price > VWAP
            if (rsi_val < 50 or price > vwap_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_RSI_MeanReversion_VWAP_Volume"
timeframe = "12h"
leverage = 1.0