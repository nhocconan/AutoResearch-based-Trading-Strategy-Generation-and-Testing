#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_TrendVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Time-based filter: 8-20 UTC only
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous day's Camarilla levels (R1, S1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    range_hl = prev_high - prev_low
    r1 = prev_close + 1.1 * range_hl / 12  # Resistance 1
    s1 = prev_close - 1.1 * range_hl / 12  # Support 1
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and 4h uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and 4h downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend and volume confirmation
# - Camarilla R1/S1 from previous day act as intraday support/resistance
# - Breakout above S1 with volume in 4h uptrend = long opportunity
# - Breakdown below R1 with volume in 4h downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise from low-volume hours
# - Position size 0.20 limits drawdown from adverse moves
# - Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend) markets
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - 4h trend filter reduces whipsaws vs using same timeframe
# - Novel combination: Camarilla (1d) + trend (4h) + volume (1h) on 1h timeframe