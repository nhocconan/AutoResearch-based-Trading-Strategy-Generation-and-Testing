#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for signal generation with Camarilla pivot breakouts
# 1w EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (2.0x 20-period average) filters for strong participation to reduce false breakouts
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by only trading in direction of 1w trend
# Camarilla provides objective price levels, reducing subjectivity in entries/exits
# Using R4/S4 levels for stronger breakout confirmation vs R3/S3

name = "12h_Camarilla_R4S4_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for Camarilla pivots (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d data for Camarilla pivots (based on previous 1d bar)
    df_1d_prev_close = df_1d['close'].shift(1)
    df_1d_prev_high = df_1d['high'].shift(1)
    df_1d_prev_low = df_1d['low'].shift(1)
    
    # Camarilla levels: based on previous 1d bar's range
    camarilla_range = df_1d_prev_high - df_1d_prev_low
    camarilla_r4 = df_1d_prev_close + camarilla_range * 1.1 / 2  # R4 = C + 1.1*range/2
    camarilla_s4 = df_1d_prev_close - camarilla_range * 1.1 / 2  # S4 = C - 1.1*range/2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R4 + volume spike + price > 1w EMA50
            if close[i] > camarilla_r4_aligned[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S4 + volume spike + price < 1w EMA50
            elif close[i] < camarilla_s4_aligned[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla S4 or price < 1w EMA50
            if close[i] < camarilla_s4_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla R4 or price > 1w EMA50
            if close[i] > camarilla_r4_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals