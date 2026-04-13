#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot (H3/L3) breakout with 1w EMA200 trend filter + volume confirmation
    # Long: price > H3 + price > 1w EMA200 + volume > 2.0x 20-period average
    # Short: price < L3 + price < 1w EMA200 + volume > 2.0x 20-period average
    # Exit: opposite pivot level breakout OR price crosses 1w EMA200
    # Using 12h timeframe to reduce trade frequency, 1w EMA200 for strong trend filter,
    # and volume spike confirmation to avoid false breakouts in choppy markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 with min_periods
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])  # SMA200 as seed
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) for each 1d bar
    camarilla_H3 = np.full(len(close_1d), np.nan)
    camarilla_L3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day's data
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            range_val = H - L
            camarilla_H3[i] = C + range_val * 1.1 / 4
            camarilla_L3[i] = C - range_val * 1.1 / 4
    
    # Align 1w EMA200 to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Align 1d Camarilla levels to 12h
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_H3_aligned[i]
        short_breakout = close[i] < camarilla_L3_aligned[i]
        
        # Trend filter from 1w EMA200
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation (>2.0x 20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma)
        else:
            volume_spike = False
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike
        short_entry = short_breakout and bearish_trend and volume_spike
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_1w_aligned[i])
        short_exit = long_breakout or (close[i] > ema_1w_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "12h_1w_camarilla_breakout_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0