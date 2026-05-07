#!/usr/bin/env python3
name = "1h_4h_1d_PivotBreakout_TrendVolume_v1"
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
    
    # Session filter: 08-20 UTC (only trade during active hours)
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Pivot from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Pivot support/resistance levels (S1, R1)
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align daily levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and 4h uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.8
            uptrend = ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]
            
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

# Hypothesis: 1h Pivot S1/R1 breakout with 4h trend and volume confirmation
# - Daily Pivot S1/R1 act as key support/resistance levels from prior session
# - Breakout above S1 with volume in 4h uptrend = long opportunity
# - Breakdown below R1 with volume in 4h downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise trades during low-liquidity hours
# - Position size 0.20 targets ~15-35 trades/year, avoiding fee drag
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Uses 4h trend filter to reduce whipsaws vs same-timeframe signals
# - Novel combination: Pivot (1d) + trend (4h) + volume (1h) with session filter on 1h timeframe
# - Aims for 60-140 total trades over 4 years (15-35/year) to stay within limits for 1h timeframe