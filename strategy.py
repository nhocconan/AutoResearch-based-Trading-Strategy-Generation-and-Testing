#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: On 1h timeframe, use 4h Camarilla R1/S1 breakouts with 4h trend filter (EMA50) and 1d volume spike confirmation.
Goes long when price breaks above R1 with 4h uptrend and 1d volume > 2.0x 20-period average,
short when price breaks below S1 with 4h downtrend and 1d volume > 2.0x 20-period average.
Uses discrete sizing (0.20) to minimize fees. Target: 15-37 trades/year (60-150 over 4 years).
Uses 4h/1d for signal direction, 1h only for entry timing. Session filter 08-20 UTC to reduce noise.
Works in bull via breakouts with trend, in bear via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculations and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for today using yesterday's 4h OHLC
    prev_close = np.concatenate([[close_4h[0]], close_4h[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_4h[0]], high_4h[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_4h[0]], low_4h[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.275 * camarilla_range
    s1 = prev_close - 0.275 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend (price > EMA50), 1d volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema_50_4h_aligned[i]) and (vol_spike_1d_aligned[i] > 0.5)
            # Short: price breaks below S1, 4h downtrend (price < EMA50), 1d volume spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema_50_4h_aligned[i]) and (vol_spike_1d_aligned[i] > 0.5)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price closes below S1 (mean reversion) or 4h trend turns down
            exit_signal = (close[i] < s1_aligned[i]) or (close[i] < ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price closes above R1 (mean reversion) or 4h trend turns up
            exit_signal = (close[i] > r1_aligned[i]) or (close[i] > ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0