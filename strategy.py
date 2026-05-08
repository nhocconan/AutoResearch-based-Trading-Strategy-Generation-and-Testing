#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h with 4h trend filter and 1d volume confirmation
# Uses 4h EMA20 for trend direction, 1d volume spike for confirmation
# 1h for precise entry timing with session filter (08-20 UTC)
# Target: 15-37 trades/year (~60-150 total over 4 years) to minimize fee drag
# Works in both bull/bear markets by filtering for strong trends with volume confirmation

name = "1h_EMA20_4hTrend_1dVol_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume spike detection (volume > 2x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 4h EMA20 (uptrend) + volume spike
            if close[i] > ema_20_4h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA20 (downtrend) + volume spike
            elif close[i] < ema_20_4h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA20 or session ends
            if close[i] < ema_20_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA20 or session ends
            if close[i] > ema_20_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals