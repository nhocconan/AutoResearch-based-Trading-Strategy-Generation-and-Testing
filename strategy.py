#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h high/low for Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Camarilla R1 and S1 from previous 4h bar
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_r1 = df_4h['close'].values + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = df_4h['close'].values - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h trend filter: EMA50
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, volume spike
            long_cond = (close[i] > camarilla_r1_aligned[i] and
                        ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and
                        volume_spike_1d_aligned[i] > 0.5)
            
            # Short: price breaks below S1, 4h downtrend, volume spike
            short_cond = (close[i] < camarilla_s1_aligned[i] and
                         ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and
                         volume_spike_1d_aligned[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or 4h trend turns down
            if (close[i] < camarilla_s1_aligned[i] or
                ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above R1 or 4h trend turns up
            if (close[i] > camarilla_r1_aligned[i] or
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals