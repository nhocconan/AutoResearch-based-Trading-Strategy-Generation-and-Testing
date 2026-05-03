#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + volume confirmation + 1d EMA(34) trend filter
# Long when price breaks above Camarilla R3 + volume spike + price > 1d EMA(34)
# Short when price breaks below Camarilla S3 + volume spike + price < 1d EMA(34)
# Uses Camarilla pivot levels from 1d for institutional support/resistance and 1d EMA(34) for trend alignment
# Designed for low trade frequency (12-37/year on 12h) to minimize fee drag. Works in both bull and bear markets via trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d: H, L, C from previous day
    # Camarilla R3 = C + (H-L) * 1.1/2
    # Camarilla S3 = C - (H-L) * 1.1/2
    # Note: Using previous day's HLC to avoid look-ahead
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_h = np.roll(h_1d, 1)
    prev_l = np.roll(l_1d, 1)
    prev_c = np.roll(c_1d, 1)
    # First value will be invalid (rolled from last), but min_periods in alignment handles this
    rng = prev_h - prev_l
    camarilla_r3 = prev_c + rng * 1.1 / 2.0
    camarilla_s3 = prev_c - rng * 1.1 / 2.0
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(20 for volume MA, 34 for 1d EMA) + 1 for shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + volume spike + price > 1d EMA(34)
            if (close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 + volume spike + price < 1d EMA(34)
            elif (close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price below 1d EMA(34)
            if (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price above 1d EMA(34)
            if (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals