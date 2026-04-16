#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d EMA filter and volume confirmation
# Uses 1d Camarilla levels (R1/S1 for breakout, R3/S3 for reversal) with 1d EMA34 trend filter
# Long when price breaks above R1 with EMA34 up and volume > 1.5x average
# Short when price breaks below S1 with EMA34 down and volume > 1.5x average
# Avoids overtrading by requiring confluence of pivot, trend, and volume
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    r3_1d = pivot_1d + range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 4
    r4_1d = pivot_1d + range_1d * 1.1 / 2
    s4_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA34 Trend Filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d Volume Confirmation (average volume) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema = ema_34_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_ma * 1.5
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if price drops below S3 (reversal zone)
            if price < s3:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit if price rises above R3 (reversal zone)
            if price > r3:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above R1 AND EMA34 up AND volume confirmation
            if price > r1 and ema > close_1d[-1] if len(close_1d) > 0 else False and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below S1 AND EMA34 down AND volume confirmation
            elif price < s1 and ema < close_1d[-1] if len(close_1d) > 0 else False and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1_S1_1dEMA34_Volume1.5x"
timeframe = "6h"
leverage = 1.0