#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1d volume confirmation for entry timing.
# Enters long when price crosses above Supertrend (uptrend) with 1d volume spike, short when price crosses below Supertrend (downtrend) with volume spike.
# Exits when price crosses back over Supertrend. Uses 4h for trend (reduces whipsaw) and 1d volume for confirmation (avoids low-volume noise).
# Designed to work in both bull and bear markets by following the 4h trend. Target: 15-35 trades/year to minimize fee drag on 1h timeframe.

name = "1h_Supertrend_4h_1dVolume"
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
    
    # Get 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate ATR(10) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend (10, 3.0) on 4h
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    supertrend = np.zeros_like(close_4h)
    dir_ = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    dir_[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
        
        if dir_[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    dir_aligned = align_htf_to_ltf(prices, df_4h, dir_)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Volume spike: current volume > 2.0 * 20-period average
    volume_spike = volume > (vol_ma_20_aligned * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for Supertrend and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(dir_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st = supertrend_aligned[i]
        direction = dir_aligned[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price crosses above Supertrend (uptrend) + volume spike
            if close[i] > st and direction == 1 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: Price crosses below Supertrend (downtrend) + volume spike
            elif close[i] < st and direction == -1 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Supertrend
            if close[i] < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above Supertrend
            if close[i] > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals