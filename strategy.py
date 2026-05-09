#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (standard)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Previous day's values for today's Camarilla levels
    prev_high_d = np.roll(high_d, 1)
    prev_low_d = np.roll(low_d, 1)
    prev_close_d = np.roll(close_d, 1)
    prev_high_d[0] = np.nan
    prev_low_d[0] = np.nan
    prev_close_d[0] = np.nan
    
    # Camarilla formula: Range = H - L
    range_d = prev_high_d - prev_low_d
    # Levels
    r3_d = prev_close_d + range_d * 1.1 / 4
    s3_d = prev_close_d - range_d * 1.1 / 4
    r4_d = prev_close_d + range_d * 1.1 / 2
    s4_d = prev_close_d - range_d * 1.1 / 2
    
    # Align daily Camarilla levels to 6h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)
    r4_d_aligned = align_htf_to_ltf(prices, df_d, r4_d)
    s4_d_aligned = align_htf_to_ltf(prices, df_d, s4_d)
    
    # Daily trend filter: 34-period EMA
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA34 (daily) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_d_aligned[i]) or 
            np.isnan(r3_d_aligned[i]) or
            np.isnan(s3_d_aligned[i]) or
            np.isnan(r4_d_aligned[i]) or
            np.isnan(s4_d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_d_val = ema34_d_aligned[i]
        r3_d_val = r3_d_aligned[i]
        s3_d_val = s3_d_aligned[i]
        r4_d_val = r4_d_aligned[i]
        s4_d_val = s4_d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Break above R4 with uptrend and volume spike
            if close[i] > r4_d_val and close[i] > ema34_d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Break below S4 with downtrend and volume spike
            elif close[i] < s4_d_val and close[i] < ema34_d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below R3 (mean reversion) or trend turns down
            if close[i] < r3_d_val or close[i] < ema34_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above S3 (mean reversion) or trend turns up
            if close[i] > s3_d_val or close[i] > ema34_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals