#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, trade long when price breaks above Camarilla R1 level and short when breaks below S1 level, 
filtered by 12h EMA50 trend and volume spike. Camarilla levels provide intraday support/resistance derived from prior day's range.
12h EMA50 acts as higher-timeframe trend filter. Volume spike confirms institutional participation. 
Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.30) to minimize fee drag.
Works in bull/bear markets via 12h trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (based on prior 1d bar's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for prior 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - prev_close_1d), np.abs(low_1d - prev_close_1d)))
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low)
    # S1 = close - 0.5*(high-low), S2 = close - 0.75*(high-low), S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + 0.5 * hl_range_1d
    s1_1d = close_1d - 0.5 * hl_range_1d
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA50 (50), volume MA (20), and Camarilla needs 1d data
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_50_val = ema_50_12h_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA50, volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below S1, below 12h EMA50, volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below S1 OR below 12h EMA50
            if (close_val < s1_val) or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above R1 OR above 12h EMA50
            if (close_val > r1_val) or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0