#!/usr/bin/env python3
# 4h_Camarilla_Pivot_S1_S2_Breakout_12hTrend_VolumeConfirm
# Hypothesis: Camarilla pivot S1/S2 breakout with 12h EMA trend filter and volume confirmation.
# Camarilla levels (S1/S2) act as support/resistance zones. Breakouts signal continuation.
# 12h EMA ensures alignment with intermediate trend. Volume spike confirms breakout strength.
# Designed for 20-40 trades/year to minimize fee drag and work in bull/bear markets.

name = "4h_Camarilla_Pivot_S1_S2_Breakout_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Use daily high/low/close for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    # Range = H - L
    range_hl = prev_high_4h - prev_low_4h
    
    # S1 = C - (H - L) * 1.0833
    # S2 = C - (H - L) * 1.1666
    s1 = prev_close_4h - range_hl * 1.0833
    s2 = prev_close_4h - range_hl * 1.1666
    
    # R1 = C + (H - L) * 1.0833
    # R2 = C + (H - L) * 1.1666
    r1 = prev_close_4h + range_hl * 1.0833
    r2 = prev_close_4h + range_hl * 1.1666
    
    # 12h EMA for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for 12h EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(s1[i]) or np.isnan(s2[i]) or np.isnan(r1[i]) or np.isnan(r2[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 with uptrend and volume
            if close[i] > r1[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with downtrend and volume
            elif close[i] < s1[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 or trend reversal
            if close[i] < s1[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or trend reversal
            if close[i] > r1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals