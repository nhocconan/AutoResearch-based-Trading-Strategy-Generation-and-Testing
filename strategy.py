#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d RSI momentum and volume confirmation.
# Long: RSI(14) > 60 on 1d + price > 20-period SMA on 4h + volume > 1.5x average volume.
# Short: RSI(14) < 40 on 1d + price < 20-period SMA on 4h + volume > 1.5x average volume.
# Uses 1d RSI for momentum bias and 4h SMA for trend filter. Volume confirms strength.
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
# Works in bull (momentum continuation) and bear (mean reversion at extremes).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for RSI
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
    
    # Initialize first average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Wilder smoothing
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs))
        else:
            rsi_1d[i] = 100
    
    # 4h SMA(20) for trend filter
    sma_20 = np.full(n, np.nan)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        sma = sma_20[i]
        rsi = rsi_1d_aligned[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: RSI > 60 (bullish momentum) + price > SMA + volume confirmation
            if (rsi > 60 and price > sma and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI < 40 (bearish momentum) + price < SMA + volume confirmation
            elif (rsi < 40 and price < sma and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI < 50 (momentum fading) or price < SMA
            if (rsi < 50 or price < sma):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI > 50 (momentum fading) or price > SMA
            if (rsi > 50 or price > sma):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0