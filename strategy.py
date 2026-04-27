#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA200_Trend_VolumeSpike_SessionFilter
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA200 trend filter and volume spike. Uses 4h for signal direction, 1h for entry timing. Session filter (08-20 UTC) reduces noise trades. Designed for 15-35 trades/year on BTC/ETH/SOL. Works in bull (breakouts with volume + trend) and bear (fade false breakouts, trend filter prevents wrong-way trades). Position size 0.20 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    # prices['open_time'] is already datetime64[ms]
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA200 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 4.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(200, 20)  # EMA200, volume avg
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_4h_aligned[i]
        size = 0.20  # 20% position size
        
        if position == 0:
            # Flat - look for breakout in direction of 4h trend with volume confirmation
            # Long: price above 4h EMA200 AND break above R1 + volume spike
            long_entry = (close_val > ema_trend) and (close_val > R1_aligned[i]) and volume_spike[i]
            # Short: price below 4h EMA200 AND break below S1 + volume spike
            short_entry = (close_val < ema_trend) and (close_val < S1_aligned[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement
            if close_val < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement
            if close_val > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA200_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0