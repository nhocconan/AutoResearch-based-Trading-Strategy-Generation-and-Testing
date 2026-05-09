#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBear_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for EMA13 (1d) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_1d_val = ema13_1d_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_spike = volume_spike[i]
        in_session = session_mask[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (strength) + 1d uptrend + volume spike + session
            if bull_power_val > 0 and close[i] > ema13_1d_val and vol_spike and in_session:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (weakness) + 1d downtrend + volume spike + session
            elif bear_power_val < 0 and close[i] < ema13_1d_val and vol_spike and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or 1d trend turns down
            if bull_power_val <= 0 or close[i] < ema13_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive or 1d trend turns up
            if bear_power_val >= 0 or close[i] > ema13_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals