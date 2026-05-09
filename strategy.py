#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day high/low as breakout levels with 1-week trend filter and volume confirmation.
# Enters long when price breaks above 1-day high with weekly uptrend and volume spike, short when price breaks below 1-day low with weekly downtrend and volume spike.
# Exits on trend reversal or price crossing opposite level (high<->low). Uses weekly timeframe for trend to avoid look-ahead and capture longer-term bias.
# Designed to work in both bull and bear markets by aligning with weekly trend. Target: 20-50 trades/year to minimize fee drag.

name = "4h_DailyHighLow_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get 1d data for high/low breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous 1d high and low as breakout levels (to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d high/low to 4h timeframe (shifted by 1 day to use completed bar)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d, additional_delay_bars=0)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d, additional_delay_bars=0)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA20 (1w) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        high_level = high_1d_aligned[i]
        low_level = low_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above 1d high + 1w uptrend + volume spike
            if close[i] > high_level and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below 1d low + 1w downtrend + volume spike
            elif close[i] < low_level and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below 1d low or 1w trend turns down
            if close[i] < low_level or close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above 1d high or 1w trend turns up
            if close[i] > high_level or close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals