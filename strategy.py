#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA20 trend filter and 1d volume spike confirmation produce moderate-frequency, high-quality trades. Uses 4h for trend direction, 1d for regime-filtered volume confirmation, and 1h for precise entry timing. Designed for 1h timeframe with session filter (08-20 UTC) to reduce noise. Target: 60-150 total trades over 4 years (15-37/year). Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with volume confirmation.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA20 for regime filter
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_R1 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_S1 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1h timeframe (completed 1d bars only)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 20 for volume MA)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if not (8 <= hours[i] <= 20):
            # Outside session: go flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            # Hold current position
            signals[i] = 0.2 if position == 1 else (-0.2 if position == -1 else 0.0)
            continue
        
        # Volume spike condition: current 1h volume > 1.5x 1d average volume (per bar)
        # Scale 1d average volume to 1h equivalent: 1d vol / 24 (approx)
        vol_1h_equiv = vol_ma20_1d_aligned[i] / 24.0
        volume_spike = volume[i] > 1.5 * vol_1h_equiv
        
        # Camarilla R1/S1 breakout conditions
        breakout_up = close[i] > R1_aligned[i]   # Price breaks above R1
        breakout_down = close[i] < S1_aligned[i]  # Price breaks below S1
        
        # 4h EMA20 trend filter
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        if breakout_up and uptrend and volume_spike:
            # Long signal: break above R1 + uptrend + volume spike
            if position != 1:
                signals[i] = 0.2
                position = 1
            else:
                signals[i] = 0.2
        elif breakout_down and downtrend and volume_spike:
            # Short signal: break below S1 + downtrend + volume spike
            if position != -1:
                signals[i] = -0.2
                position = -1
            else:
                signals[i] = -0.2
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.2
            else:
                signals[i] = -0.2
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0