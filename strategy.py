#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h trend filter and volume confirmation
# Uses Williams %R to identify overbought/oversold conditions, filtered by 4h EMA50 trend
# Volume spike required for entry to avoid false signals
# Designed to work in both bull and bear markets by following higher timeframe trend
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

name = "1h_WilliamsR_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Williams %R(14) on 1h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: oversold + uptrend + volume spike
            if (wr < -80 and 
                close[i] > ema50_4h_val and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: overbought + downtrend + volume spike
            elif (wr > -20 and 
                  close[i] < ema50_4h_val and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: overbought OR trend turns down
            if (wr > -20 or close[i] < ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: oversold OR trend turns up
            if (wr < -80 or close[i] > ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals