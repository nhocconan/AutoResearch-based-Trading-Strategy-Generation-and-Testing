#!/usr/bin/env python3
"""
Hypothesis: 6-hour Camarilla pivot + 1-day EMA trend + volume confirmation.
In bull market (1d close > 1d EMA50): long on bounce off S1/S2 or break above R3.
In bear market (1d close < 1d EMA50): short on bounce off R1/R2 or break below S3.
Volume must be above 20-period average to confirm.
Uses 6h bars for entry timing, 1d for trend and pivot levels.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_high = df_1d['high'].values
    one_d_low = df_1d['low'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === 1D CAMARILLA PIVOTS (HTF) ===
    # Classic Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    camarilla_r4 = one_d_close + (one_d_high - one_d_low) * 1.1 / 2
    camarilla_r3 = one_d_close + (one_d_high - one_d_low) * 1.1 / 4
    camarilla_r2 = one_d_close + (one_d_high - one_d_low) * 1.1 / 6
    camarilla_r1 = one_d_close + (one_d_high - one_d_low) * 1.1 / 12
    camarilla_s1 = one_d_close - (one_d_high - one_d_low) * 1.1 / 12
    camarilla_s2 = one_d_close - (one_d_high - one_d_low) * 1.1 / 6
    camarilla_s3 = one_d_close - (one_d_high - one_d_low) * 1.1 / 4
    camarilla_s4 = one_d_close - (one_d_high - one_d_low) * 1.1 / 2
    
    # Align pivots to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns bearish
            if close[i] < s3_aligned[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns bullish
            if close[i] > r3_aligned[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend and Camarilla levels
            if bull_trend:
                # In bull market: long on bounce from S1/S2 or break above R3
                if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]) or \
                   (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i]) or \
                   (high[i] >= r3_aligned[i] and close[i] > r3_aligned[i]):
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on bounce from R1/R2 or break below S3
                if (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or \
                   (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i]) or \
                   (low[i] <= s3_aligned[i] and close[i] < s3_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals