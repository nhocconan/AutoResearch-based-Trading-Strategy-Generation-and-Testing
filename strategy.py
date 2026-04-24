#!/usr/bin/env python3
"""
Hypothesis: 6h Volume Spike + 1d Camarilla Pivot Breakout with Trend Filter.
- Primary timeframe: 6h for execution, HTF: 1d for Camarilla pivot levels and trend filter.
- Entry: 6h close breaks above R3 or below S3 from prior 1d Camarilla calculation + volume spike (>2.0x 20-period volume MA).
- Direction filter: only long when 6h close > 1d EMA50, only short when 6h close < 1d EMA50.
- Volume confirmation reduces false breakouts; Camarilla R3/S3 provides meaningful support/resistance.
- Exit: opposite Camarilla level touch (long exits at S3, short exits at R3) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via breakout continuation, in bear via mean reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate prior 1d Camarilla levels (using previous day's OHLC)
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    #          H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # We use R3=H3, S3=L3 for entry; R4=H4, S4=L4 for stronger breakouts
    prev_high = np.roll(close_1d, 1)  # Simplified: use prior close as proxy for prior day's high/low
    prev_low = np.roll(close_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Better approximation: use rolling window for true prior day OHLC
    # Since we don't have intraday 1d OHLC in 6h data, we approximate using 1d close
    # For proper Camarilla we need 1d OHLC - we'll use 1d close as base and estimate range
    range_1d = pd.Series(close_1d).rolling(window=2, min_periods=1).std() * np.sqrt(2) * 2  # Rough range estimate
    prev_range = np.roll(range_1d.values, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels based on prior 1d close and estimated range
    r3 = prev_close_1d + prev_range * 1.1 / 4
    s3 = prev_close_1d - prev_range * 1.1 / 4
    r4 = prev_close_1d + prev_range * 1.1 / 2
    s4 = prev_close_1d - prev_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 OR R4 with volume spike AND uptrend
            if ((close[i] > r3_aligned[i] or close[i] > r4_aligned[i]) and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 OR S4 with volume spike AND downtrend
            elif ((close[i] < s3_aligned[i] or close[i] < s4_aligned[i]) and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below S3 (mean reversion) or trend reversal
            if close[i] < s3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above R3 (mean reversion) or trend reversal
            if close[i] > r3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_VolumeSpike_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0