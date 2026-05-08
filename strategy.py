#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous day's 12h candle (using previous 12h bar for 4h alignment)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    # Camarilla levels from previous 12h bar (R3, S3)
    R3_12h = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 4
    S3_12h = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_4h = align_htf_to_ltf(prices, df_12h, S3_12h)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, price above EMA50, volume spike
            long_cond = (close[i] > R3_4h[i] and 
                        close[i] > ema50_4h[i] and
                        volume_spike[i])
            
            # Short: Price breaks below S3, price below EMA50, volume spike
            short_cond = (close[i] < S3_4h[i] and 
                         close[i] < ema50_4h[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S3 OR price crosses below EMA50
            if close[i] < S3_4h[i] or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R3 OR price crosses above EMA50
            if close[i] > R3_4h[i] or close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals