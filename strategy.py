#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with daily EMA34 trend filter and volume spike confirmation. 
Works in bull/bear by requiring alignment with daily trend. Discrete sizing (0.25) minimizes fee drag.
Target: 75-200 trades over 4 years. Uses price channel structure proven effective on ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for daily EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels (based on previous day's OHLC)
    # Calculate daily OHLC from 4h data
    # We'll use rolling window of 6 bars (6*4h = 24h) to approximate daily
    period = 6
    if n < period:
        return np.zeros(n)
    
    # Rolling window for daily high/low/close
    roll_high = pd.Series(high).rolling(window=period, min_periods=period)
    roll_low = pd.Series(low).rolling(window=period, min_periods=period)
    roll_close = pd.Series(close).rolling(window=period, min_periods=period)
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = roll_high.max().shift(1).values
    prev_low = roll_low.min().shift(1).values
    prev_close = roll_close.mean().shift(1).values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    camarilla_r4 = prev_close + range_val * 1.1 / 2
    camarilla_s4 = prev_close - range_val * 1.1 / 2
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 6 for Camarilla + 34 for EMA)
    start_idx = max(period + 1, 34)  # +1 for shift
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 with volume spike and daily uptrend
        long_condition = (close[i] > camarilla_r3[i]) and volume_spike[i] and (close[i] > ema_34_1d_aligned[i])
        # Short logic: price breaks below S3 with volume spike and daily downtrend
        short_condition = (close[i] < camarilla_s3[i]) and volume_spike[i] and (close[i] < ema_34_1d_aligned[i])
        
        # Exit logic: price re-enters R3/S3 range or daily trend reversal
        exit_long = close[i] < camarilla_r3[i] or close[i] < ema_34_1d_aligned[i]
        exit_short = close[i] > camarilla_s3[i] or close[i] > ema_34_1d_aligned[i]
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0