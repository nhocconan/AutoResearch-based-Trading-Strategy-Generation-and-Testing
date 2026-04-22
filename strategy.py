#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once (no look-ahead)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's HLC for pivot calculation (no look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot levels (standard formula)
    pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pp_1w - prev_low_1w  # R1 = 2*P - Low
    s1_1w = 2 * pp_1w - prev_high_1w # S1 = 2*P - High
    
    # 12h EMA200 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema200_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema200 = ema200_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume + above 12h EMA200
            if price > r1 and vol > 2.0 * vol_ma and price > ema200:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly S1 with volume + below 12h EMA200
            elif price < s1 and vol > 2.0 * vol_ma and price < ema200:
                signals[i] = -0.30
                position = -1
        
        elif position != 0:
            # Exit: price returns to weekly central pivot (mean reversion)
            # Calculate weekly PP for exit signal
            pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
            pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
            pp = pp_aligned[i]
            
            if position == 1 and price < pp:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > pp:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_12hEMA200_Volume_Spike"
timeframe = "12h"
leverage = 1.0