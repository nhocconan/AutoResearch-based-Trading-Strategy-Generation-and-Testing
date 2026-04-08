#!/usr/bin/env python3
# 6h_camarilla_pivot_12h_trend_volume_v1
# Hypothesis: Camarilla pivot levels from 12h timeframe with trend filter from 1d EMA200 and volume confirmation.
# Long when price breaks above R3 with price above 1d EMA200 and volume > 1.5x average.
# Short when price breaks below S3 with price below 1d EMA200 and volume > 1.5x average.
# Exit on opposite signal or when price retests the pivot level (PP).
# Target: 80-120 total trades over 4 years (~20-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar high, low, close
    ph = df_12h['high'].values[:-1]  # previous bar
    pl = df_12h['low'].values[:-1]
    pc = df_12h['close'].values[:-1]
    
    # Camarilla calculations
    range_val = ph - pl
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    r3 = pc + range_val * 1.1 / 4
    s3 = pc - range_val * 1.1 / 4
    pp = (ph + pl + pc) / 3  # Pivot point
    
    # Align to 6h timeframe (previous 12h bar values)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    pp_6h = align_htf_to_ltf(prices, df_12h, pp)
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_6h = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(pp_6h[i]) or 
            np.isnan(ema_200_6h[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retests PP level or reverse signal
            if close[i] <= pp_6h[i] or close[i] < s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests PP level or reverse signal
            if close[i] >= pp_6h[i] or close[i] > r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long: break above R3 with uptrend
            if close[i] > r3_6h[i] and close[i] > ema_200_6h[i] and volume_ok:
                # Additional confirmation: previous close was at or below R3
                if i > 0 and close[i-1] <= r3_6h[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Short: break below S3 with downtrend
            elif close[i] < s3_6h[i] and close[i] < ema_200_6h[i] and volume_ok:
                # Additional confirmation: previous close was at or above S3
                if i > 0 and close[i-1] >= s3_6h[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals