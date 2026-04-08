#!/usr/bin/env python3
# 6h_12h_1d_camarilla_volume_v1
# Hypothesis: Use daily Camarilla pivot levels for 6h entries with volume confirmation.
# Fade at R3/S3 levels (mean reversion) when price is outside daily range but showing exhaustion.
# Breakout continuation at R4/S4 levels when price breaks with strong volume.
# Uses 12h trend filter to avoid counter-trend trades in strong trends.
# Works in bull/bear by adapting to price action at key institutional levels.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_12h_1d_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # R2 = close + 0.5 * (high - low)
    # R1 = close + 0.25 * (high - low)
    # S1 = close - 0.25 * (high - low)
    # S2 = close - 0.5 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    r4 = close_1d + 1.5 * range_1d
    r3 = close_1d + 1.0 * range_1d
    s3 = close_1d - 1.0 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align daily levels to 6h
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Simple trend: price above/below 20-period EMA
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_12h = np.where(close_12h > ema20_12h, 1, -1)
    trend_12h_6h = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6 days
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.8  # 80% above average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 24  # Need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(trend_12h_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            # 1. Price reaches R3 (take profit)
            # 2. 12h trend turns bearish
            # 3. Price breaks below S3 (stop)
            if (close[i] >= r3_6h[i] or 
                trend_12h_6h[i] == -1 or 
                close[i] <= s3_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            # 1. Price reaches S3 (take profit)
            # 2. 12h trend turns bullish
            # 3. Price breaks above R3 (stop)
            if (close[i] <= s3_6h[i] or 
                trend_12h_6h[i] == 1 or 
                close[i] >= r3_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 levels (mean reversion)
            # Only when 12h trend is weak or aligning with fade
            if (abs(trend_12h_6h[i]) <= 1):  # Allow counter-trend if trend not strong
                # Long when price touches S3 with volume spike (bounce)
                if (close[i] <= s3_6h[i] * 1.002 and  # Allow small buffer
                    vol_spike[i]):
                    position = 1
                    signals[i] = 0.25
                # Short when price touches R3 with volume spike (rejection)
                elif (close[i] >= r3_6h[i] * 0.998 and  # Allow small buffer
                      vol_spike[i]):
                    position = -1
                    signals[i] = -0.25
            # Breakout at R4/S4 levels with volume (continuation)
            elif vol_spike[i]:
                # Long breakout above R4
                if close[i] >= r4_6h[i] * 1.001:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown below S4
                elif close[i] <= s4_6h[i] * 0.999:
                    position = -1
                    signals[i] = -0.25
    
    return signals