#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla pivot levels from 1d with volume confirmation and 12h trend filter
# Camarilla levels: Pivot = (H+L+C)/3; R1 = C + (H-L)*1.1/12; S1 = C - (H-L)*1.1/12
# In bull markets: buy at S1/S2 with upward 12h trend; sell at R3/R4
# In bear markets: sell at R1/R2 with downward 12h trend; buy at S3/S4
# Target: 15-30 trades/year, low frequency to minimize fee drag
name = "6h_camarilla_pivot_1d_volume_12h_trend_v1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H+L+C)/3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Range = H - L
    range_hl = df_1d['high'] - df_1d['low']
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    r1 = df_1d['close'] + range_hl * 1.1 / 12.0
    r2 = df_1d['close'] + range_hl * 1.1 / 6.0
    r3 = df_1d['close'] + range_hl * 1.1 / 4.0
    r4 = df_1d['close'] + range_hl * 1.1 / 2.0
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    s1 = df_1d['close'] - range_hl * 1.1 / 12.0
    s2 = df_1d['close'] - range_hl * 1.1 / 6.0
    s3 = df_1d['close'] - range_hl * 1.1 / 4.0
    s4 = df_1d['close'] - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend direction
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get daily volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: 12h EMA direction
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (strong resistance) OR trend turns down
            if close[i] >= r3_aligned[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches S3 (strong support) OR trend turns up
            if close[i] <= s3_aligned[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price at S1/S2 support + volume confirmation + uptrend
            if ((close[i] <= s1_aligned[i] * 1.001 and close[i] >= s1_aligned[i] * 0.999) or
                (close[i] <= s2_aligned[i] * 1.001 and close[i] >= s2_aligned[i] * 0.999)) and \
               vol_confirm and uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price at R1/R2 resistance + volume confirmation + downtrend
            elif ((close[i] >= r1_aligned[i] * 0.999 and close[i] <= r1_aligned[i] * 1.001) or
                  (close[i] >= r2_aligned[i] * 0.999 and close[i] <= r2_aligned[i] * 1.001)) and \
                 vol_confirm and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals