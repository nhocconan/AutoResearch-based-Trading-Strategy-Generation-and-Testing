#!/usr/bin/env python3
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
    
    # === 4h Williams %R (14-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(high_4h, np.nan)
    lowest_low = np.full_like(low_4h, np.nan)
    period = 14
    for i in range(len(high_4h)):
        if i >= period - 1:
            highest_high[i] = np.max(high_4h[i-period+1:i+1])
            lowest_low[i] = np.min(low_4h[i-period+1:i+1])
        elif i > 0:
            highest_high[i] = np.max(high_4h[0:i+1])
            lowest_low[i] = np.min(low_4h[0:i+1])
        else:
            highest_high[i] = high_4h[0]
            lowest_low[i] = low_4h[0]
    
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          (highest_high - close_4h) / (highest_high - lowest_low) * -100, 
                          -50)
    
    # === 1d RSI (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI using Wilder's smoothing
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing with proper seeding
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # === Align indicators to 4h timeframe ===
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h Volume confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_10 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 9:
            vol_ma_10[i] = np.mean(volume_4h[i-9:i+1])
        elif i > 0:
            vol_ma_10[i] = np.mean(volume_4h[max(0, i-4):i+1])
        else:
            vol_ma_10[i] = volume_4h[0]
    
    vol_confirm = volume_4h > vol_ma_10 * 1.5
    
    # === 4h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Williams %R < -80 (oversold) + RSI < 30 + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                rsi_1d_aligned[i] < 30 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R > -20 (overbought) + RSI > 70 + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  rsi_1d_aligned[i] > 70 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR RSI crosses above 50
            if (williams_r_aligned[i] > -50 or 
                rsi_1d_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR RSI crosses below 50
            if (williams_r_aligned[i] < -50 or 
                rsi_1d_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_RSI_OverboughtOversold_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0