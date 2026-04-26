#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend alignment and volume spike confirmation. In trending markets (4h close > 4h EMA20), take long at R1 breakout and short at S1 breakout. Uses volume > 1.5x 20-period average to confirm breakout strength. Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year by requiring tight confluence of trend, level break, and volume. Works in bull/bear via trend following logic - only trades in direction of 4h trend, avoiding counter-trend whipsaws.
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
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour  # Already DatetimeIndex
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    htf_trend = np.where(close > ema_20_4h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate previous day's high, low, close for Camarilla levels
    # Use 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for Camarilla)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long entry: price breaks above R1 with volume spike and 4h uptrend
        if close[i] > r1_aligned[i] and volume_spike[i] and htf_trend[i] == 1:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        
        # Short entry: price breaks below S1 with volume spike and 4h downtrend
        elif close[i] < s1_aligned[i] and volume_spike[i] and htf_trend[i] == -1:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        
        # Exit conditions: reverse signal or loss of trend/volume
        elif position == 1 and (close[i] < s1_aligned[i] or htf_trend[i] != 1 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r1_aligned[i] or htf_trend[i] != -1 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0