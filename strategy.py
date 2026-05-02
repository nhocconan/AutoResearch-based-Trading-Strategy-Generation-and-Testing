#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 6h primary timeframe for signal generation with Camarilla pivot breakouts
# 12h EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (2.0x 20-period average) filters for strong participation to reduce false breakouts
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by only trading in direction of 12h trend
# Camarilla provides objective price levels, reducing subjectivity in entries/exits

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h data for Camarilla pivots (based on previous 12h bar)
    df_12h_prev_close = df_12h['close'].shift(1)
    df_12h_prev_high = df_12h['high'].shift(1)
    df_12h_prev_low = df_12h['low'].shift(1)
    
    # Camarilla levels: based on previous 12h bar's range
    camarilla_range = df_12h_prev_high - df_12h_prev_low
    camarilla_r3 = df_12h_prev_close + camarilla_range * 1.1 / 4  # R3 = C + 1.1*range/4
    camarilla_s3 = df_12h_prev_close - camarilla_range * 1.1 / 4  # S3 = C - 1.1*range/4
    camarilla_r4 = df_12h_prev_close + camarilla_range * 1.1 / 2  # R4 = C + 1.1*range/2
    camarilla_s4 = df_12h_prev_close - camarilla_range * 1.1 / 2  # S4 = C - 1.1*range/2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4.values)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R3 + volume spike + price > 12h EMA50
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 + volume spike + price < 12h EMA50
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla S3 or price < 12h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla R3 or price > 12h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals