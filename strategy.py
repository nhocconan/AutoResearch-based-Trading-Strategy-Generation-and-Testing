#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend and volume confirmation
# Long when price breaks above Camarilla R3 on 4h, 1d EMA34 rising, volume > 1.5x average
# Short when price breaks below Camarilla S3 on 4h, 1d EMA34 falling, volume > 1.5x average
# Uses Camarilla levels for precise entry/exit points, 1d EMA34 for trend filter to avoid whipsaws
# Targets 30-80 total trades over 4 years (7-20/year) for low fee drag and high win rate
# Based on proven patterns from DB showing Camarilla breakouts with volume perform well on ETH/SOL

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h OHLC (previous day's range)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous period's range
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    range_4h = high_4h - low_4h
    camarilla_r3 = close_4h + range_4h * 1.1 / 2
    camarilla_s3 = close_4h - range_4h * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one bar for Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3, 1d uptrend, volume spike
            if high_val > r3_val and ema34_1d_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3, 1d downtrend, volume spike
            elif low_val < s3_val and ema34_1d_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or 1d trend down
            if low_val < s3_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or 1d trend up
            if high_val > r3_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals