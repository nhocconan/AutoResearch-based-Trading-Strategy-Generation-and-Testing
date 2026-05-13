#!/usr/bin/env python3
name = "1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
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
    
    # Calculate 1h period Camarilla levels (using previous 1h bar)
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA20 on 4h close
    ema_20_4h = np.zeros_like(close_4h)
    ema_20_4h[:] = np.nan
    if len(close_4h) >= 20:
        ema_20_4h[19] = close_4h[:20].mean()
        for i in range(20, len(close_4h)):
            ema_20_4h[i] = (close_4h[i] * 2 + ema_20_4h[i-1] * 19) / 21
    
    # Align 4h EMA20 to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate average daily volume (20-day)
    avg_vol_1d = np.zeros_like(volume_1d)
    avg_vol_1d[:] = np.nan
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            avg_vol_1d[i] = volume_1d[i-19:i+1].mean()
    
    # Align 1d average volume to 1h timeframe
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x average daily volume
        vol_spike = volume[i] > 1.5 * avg_vol_1d_aligned[i]
        
        if position == 0:
            # LONG: Close > R1 + volume spike + 4h uptrend (close > EMA20)
            if (close[i] > R1[i] and vol_spike and close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Close < S1 + volume spike + 4h downtrend (close < EMA20)
            elif (close[i] < S1[i] and vol_spike and close[i] < ema_20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S1 (reversal to opposite level)
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close > R1 (reversal to opposite level)
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals