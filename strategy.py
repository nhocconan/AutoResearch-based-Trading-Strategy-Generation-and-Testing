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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: S1, S2, S3, R1, R2, R3 (Camarilla)
    s1 = close_1d - (high_1d - low_1d) * 1.05 / 12
    s2 = close_1d - (high_1d - low_1d) * 1.10 / 6
    s3 = close_1d - (high_1d - low_1d) * 1.10 / 4
    r1 = close_1d + (high_1d - low_1d) * 1.05 / 12
    r2 = close_1d + (high_1d - low_1d) * 1.10 / 6
    r3 = close_1d + (high_1d - low_1d) * 1.10 / 4
    
    # Align levels to 12h timeframe
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume filter: 20-period volume moving average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate 12-period RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(r3_12h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi[i] > 30) and (rsi[i] < 70)
        
        # Entry conditions: Break of S1/R1 with volume and RSI filter
        long_breakout = (close[i] > r1_12h[i]) and volume_filter and rsi_filter
        short_breakout = (close[i] < s1_12h[i]) and volume_filter and rsi_filter
        
        # Exit conditions: Return to S2/R2 (mean reversion)
        long_exit = close[i] < s2_12h[i]
        short_exit = close[i] > r2_12h[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_s1r1_breakout_s2r2_reversion_v1"
timeframe = "12h"
leverage = 1.0