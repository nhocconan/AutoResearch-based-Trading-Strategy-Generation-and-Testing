#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA34 trend filter and volume confirmation.
Camarilla pivot levels (R3/S3) represent strong intraday support/resistance derived from prior day's range.
Breakouts above R3 or below S3 with volume and 1d trend alignment capture sustained moves.
The 1d EMA34 provides adaptive trend filter that works in both bull and bear markets.
Targeting 80-120 total trades over 4 years (20-30/year) to balance signal quality and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels on 1d data (using prior day's OHLC)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C, H, L are from prior day (shifted by 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high, 1) if hasattr(high, '__len__') else np.roll(df_1d['high'].values, 1)
    low_1d_shift = np.roll(low, 1) if hasattr(low, '__len__') else np.roll(df_1d['low'].values, 1)
    
    # Handle first bar (no prior day)
    close_1d_shift[0] = close_1d[0]
    high_1d_shift[0] = high_1d[0] if hasattr(high_1d, '__len__') else df_1d['high'].iloc[0]
    low_1d_shift[0] = low_1d[0] if hasattr(low_1d, '__len__') else df_1d['low'].iloc[0]
    
    camarilla_range = (high_1d_shift - low_1d_shift) * 1.1 / 4
    r3 = close_1d_shift + camarilla_range
    s3 = close_1d_shift - camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(avg_volume[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike + in uptrend
        if close[i] > r3_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S3 with volume spike + in downtrend
        elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Camarilla level or trend weakens
        elif position == 1 and (close[i] < s3_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r3_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0