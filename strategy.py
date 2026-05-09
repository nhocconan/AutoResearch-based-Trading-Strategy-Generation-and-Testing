#!/usr/bin/env python3
# 1h_Triple_Confirmation_Strategy
# Hypothesis: Use 4h trend (EMA50), 1h momentum (RSI), and volume confirmation for high-probability entries.
# In bull markets: 4h EMA50 up + RSI > 55 + volume > 1.5x average → long
# In bear markets: 4h EMA50 down + RSI < 45 + volume > 1.5x average → short
# Uses 4h for trend direction (reduces whipsaw), 1h for timing, volume for confirmation.
# Target: 15-30 trades/year to minimize fee drag in 1h timeframe.
# Works in both bull and bear by following 4h trend direction.

name = "1h_Triple_Confirmation_Strategy"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close, np.nan)
    valid = (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    
    rsi = np.full_like(close, np.nan)
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    
    # Volume ratio: current / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h uptrend + RSI > 55 + volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi[i] > 55 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + RSI < 45 + volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi[i] < 45 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or RSI overextended
            if close[i] < ema_50_4h_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or RSI oversold
            if close[i] > ema_50_4h_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals