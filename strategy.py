#!/usr/bin/env python3
name = "4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla R1 and S1 levels from previous 4h bar
    R1 = np.zeros(n)
    S1 = np.zeros(n)
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R1[i] = prev_close + range_val * 1.1 / 4
            S1[i] = prev_close - range_val * 1.1 / 4
        else:
            R1[i] = prev_close
            S1[i] = prev_close
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA50 on 12h close
    ema_50 = np.zeros_like(close_12h)
    ema_50[:] = np.nan
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_50[i] = close_12h[i]
        elif np.isnan(ema_50[i-1]):
            ema_50[i] = close_12h[i]
        else:
            ema_50[i] = alpha * close_12h[i] + (1 - alpha) * ema_50[i-1]
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Close > R1 + volume spike + 12h uptrend (close > EMA50)
            if (close[i] > R1[i] and vol_spike and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + volume spike + 12h downtrend (close < EMA50)
            elif (close[i] < S1[i] and vol_spike and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S1 (reversal to opposite level)
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > R1 (reversal to opposite level)
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals