#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Camarilla pivot breakout with 4h trend filter and volume confirmation.
# Long when price breaks above R1 with 4h EMA(50) uptrend and volume spike.
# Short when price breaks below S1 with 4h EMA(50) downtrend and volume spike.
# Uses 1h timeframe for precision entries, 4h for direction to reduce whipsaw.
# Target: 15-37 trades/year (60-150 total over 4 years) to avoid fee drag.
# Session filter (08-20 UTC) excludes low-activity periods.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla pivot levels from 4h data
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    camarilla_r1 = close_4h_vals + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h_vals - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume spike: current volume > 2.0 * 20-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # pre-computed DatetimeIndex.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + 4h uptrend + volume spike
            if (not np.isnan(r1_level) and close[i] > r1_level and 
                close[i] > ema50_4h_val and vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S1 + 4h downtrend + volume spike
            elif (not np.isnan(s1_level) and close[i] < s1_level and 
                  close[i] < ema50_4h_val and vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR 4h trend turns down
            if (not np.isnan(s1_level) and close[i] < s1_level) or close[i] < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 OR 4h trend turns up
            if (not np.isnan(r1_level) and close[i] > r1_level) or close[i] > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals