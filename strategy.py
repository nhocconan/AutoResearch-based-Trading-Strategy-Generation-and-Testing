#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike.
Long when price breaks above R1 with 4h EMA50 uptrend and volume spike (08-20 UTC).
Short when price breaks below S1 with 4h EMA50 downtrend and volume spike (08-20 UTC).
Exit on opposite band touch or trend reversal.
Uses discrete sizing (0.20) to minimize fees. Target: 15-37 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at bands.
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
    
    # Get 4h data for Camarilla calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    R4_4h = np.full(len(close_4h), np.nan)
    S4_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
            R4_4h[i] = close_prev + (range_prev * 1.1 / 2)
            S4_4h[i] = close_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R4_4h_aligned = align_htf_to_ltf(prices, df_4h, R4_4h)
    S4_4h_aligned = align_htf_to_ltf(prices, df_4h, S4_4h)
    
    # Get 4h data for trend filter (EMA50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike (session filter)
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_50_4h_aligned[i]) and vol_spike[i] and in_session[i]
            # Short: price breaks below S1 with downtrend and volume spike (session filter)
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_50_4h_aligned[i]) and vol_spike[i] and in_session[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions: price touches S1 or trend reverses
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0