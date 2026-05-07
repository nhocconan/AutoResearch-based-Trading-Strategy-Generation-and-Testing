#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h-based 1d equivalent for Camarilla pivot levels
    # Using 4h high/low/close to compute daily pivot levels
    # For each 4h bar, we compute Camarilla levels based on previous 4h bar's high/low/close
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    prev_4h_close = df_4h['close'].shift(1).values
    
    pivot = (prev_4h_high + prev_4h_low + prev_4h_close) / 3
    range_hl = prev_4h_high - prev_4h_low
    
    # Camarilla levels from 4h data (S1 and R1)
    s1_4h = prev_4h_close - (range_hl * 1.08 / 2)
    r1_4h = prev_4h_close + (range_hl * 1.08 / 2)
    
    # Align 4h Camarilla levels to 1h timeframe
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    
    # Daily trend filter: EMA(34) on daily close from 1h data
    # We'll use 1h data to compute EMA(34) on daily equivalent
    # But we need to use actual daily data for trend
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 12-period average (6 hours of 1h bars) for 1h volume
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 12)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_12[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_4h_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_4h_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_4h_aligned[i] or volume[i] < vol_ma_12[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_4h_aligned[i] or volume[i] < vol_ma_12[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with 4h/1d trend and volume confirmation
# - 4h Camarilla S1/R1 act as support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise trades
# - Position size 0.20 targets ~20-40 trades/year, avoiding fee drag
# - Uses 4h data for Camarilla levels and 1d for trend filter
# - Designed to work in BOTH bull and bear markets via trend filter
# - Higher volume threshold (2.0) to reduce false signals
# - Exit volume threshold (1.5) to allow trends to continue