#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: Use 1h timeframe for entry timing with Camarilla R1/S1 breakouts, 
filtered by 4h EMA20 trend direction and volume confirmation. 
Trades only during 08-20 UTC session to avoid low-liquidity periods.
Designed to work in both bull and bear markets by:
- Using 4h trend filter to align with higher timeframe momentum
- Volume spike confirmation reduces false breakouts
- Session filter avoids noisy off-hours price action
- Discrete 0.20 position size limits fee drag
Target: 15-35 trades/year (60-140 over 4 years) for optimal Sharpe generalization.
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
    
    # Precompute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex from Parquet
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point) - tighter levels for more precision
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume spike: current volume > 1.8 * 24-period average (more selective)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 24 for volume avg, 20 for 4h EMA
    start_idx = max(24, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        close_val = close[i]
        size = 0.20  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R1 + 4h EMA20 uptrend + volume spike
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below S1 + 4h EMA20 downtrend + volume spike
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S1 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R1 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0