#!/usr/bin/env python3
"""
6H Weekly Pivot Reversal with Volume Spike
Fade at weekly S3/R3 with volume spike and RSI divergence.
Continue at weekly S4/R4 with volume spike.
Exit when price reaches opposite pivot level or RSI reverses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI (14) for divergence detection ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === Weekly pivot points (calculated from previous week) ===
    df_1w = get_htf_data(prices, '1w')
    # Previous week's OHLC for pivot calculation
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    r2 = pp + (prev_week_high - prev_week_low)
    s2 = pp - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pp - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pp)
    r4 = prev_week_high + 3 * (pp - prev_week_low)
    s4 = prev_week_low - 3 * (prev_week_high - pp)
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    vol_spike = vol_ratio > 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if close[i] >= r3_aligned[i] or rsi[i] > 70:  # Reached resistance or overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if close[i] <= s3_aligned[i] or rsi[i] < 30:  # Reached support or oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at S3/R3 with volume spike and RSI divergence
            # Long: price at S3 with bullish RSI divergence
            if (abs(close[i] - s3_aligned[i]) < 0.001 * close[i] and  # At S3 level
                vol_spike[i] and 
                rsi[i] < 30 and  # Oversold
                i >= 2 and rsi[i] > rsi[i-2]):  # RSI making higher low (bullish div)
                position = 1
                signals[i] = 0.25
            # Short: price at R3 with bearish RSI divergence
            elif (abs(close[i] - r3_aligned[i]) < 0.001 * close[i] and  # At R3 level
                  vol_spike[i] and 
                  rsi[i] > 70 and  # Overbought
                  i >= 2 and rsi[i] < rsi[i-2]):  # RSI making lower high (bearish div)
                position = -1
                signals[i] = -0.25
            # Breakout continuation at S4/R4 with volume spike
            elif close[i] < s4_aligned[i] and vol_spike[i]:  # Break below S4
                position = -1
                signals[i] = -0.25
            elif close[i] > r4_aligned[i] and vol_spike[i]:  # Break above R4
                position = 1
                signals[i] = 0.25
    
    return signals