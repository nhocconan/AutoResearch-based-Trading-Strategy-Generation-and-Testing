#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (price > EMA20) and volume spike confirmation,
trading only during 08-20 UTC session to reduce noise. Uses 4h/1d for signal direction (trend, Camarilla levels)
and 1h only for entry timing precision. Discrete sizing (0.20) minimizes fee churn. Designed for 15-37 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at Camarilla extremes.
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
    
    # Precompute session hours (08-20 UTC) using DatetimeIndex
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculations (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.275 * camarilla_range
    s1 = prev_close - 0.275 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
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
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend (price > EMA20), volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema_20_4h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, 4h downtrend (price < EMA20), volume spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema_20_4h_aligned[i]) and vol_spike[i]
            
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
            exit_signal = (close[i] < s1_aligned[i]) or (close[i] < ema_20_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price closes above R1 (mean reversion) or 4h trend turns up
            exit_signal = (close[i] > r1_aligned[i]) or (close[i] > ema_20_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0