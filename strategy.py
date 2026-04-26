#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts aligned with 4h EMA trend filter and 1d volume confirmation capture strong trends while avoiding whipsaws. Using 4h for trend and 1d for volume reduces noise and overtrading. Session filter (08-20 UTC) further improves quality. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h data ONCE before loop for trend filter (EMA)
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_shifted + 1.1 * camarilla_range / 12
    s1 = close_1d_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for EMA and volume MA)
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(vol_ma_20_1h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA)
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: 1h volume > 2.0x 1h MA AND 1d volume > 1.5x 1d MA
        volume_spike_1h = volume[i] > 2.0 * vol_ma_20_1h[i]
        volume_spike_1d = volume_1d[-1] > 1.5 * vol_ma_20_1d_aligned[i] if len(volume_1d) > 0 else False
        # Use current 1d volume (approximate with last known value)
        volume_spike = volume_spike_1h and volume_spike_1d
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume
        if uptrend and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: breakout below S1 in downtrend with volume
        elif downtrend and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0