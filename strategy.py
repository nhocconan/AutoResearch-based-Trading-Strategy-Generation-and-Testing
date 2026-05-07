#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with volume spike and price > 1d EMA34 (uptrend).
# Short when price breaks below S3 with volume spike and price < 1d EMA34 (downtrend).
# Exit when price crosses back through the pivot point (CP) or volume drops below average.
# Target: 20-50 trades/year to avoid fee drag. Uses 1d EMA34 for trend filter to avoid counter-trend trades.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H = high, L = low, C = close
    H = prev_high
    L = prev_low
    C = prev_close
    
    # Calculate levels
    R3 = C + (H - L) * 1.1 / 2
    S3 = C - (H - L) * 1.1 / 2
    CP = (H + L + C) / 3  # Pivot point
    
    # Align to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    CP_4h = align_htf_to_ltf(prices, df_1d, CP)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average (higher threshold to reduce trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start from second bar to have previous day data
    
    for i in range(start_idx, n):
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(CP_4h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, price > 1d EMA34 (uptrend)
            long_cond = (close[i] > R3_4h[i]) and volume_spike[i] and (close[i] > ema34_1d_aligned[i])
            # Short conditions: price breaks below S3, volume spike, price < 1d EMA34 (downtrend)
            short_cond = (close[i] < S3_4h[i]) and volume_spike[i] and (close[i] < ema34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below pivot point OR volume spike ends
            if close[i] < CP_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above pivot point OR volume spike ends
            if close[i] > CP_4h[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals